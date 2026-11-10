"""Equivalence tests against recorded reference values.

The other test files check that the virtual-node transform and the forward
pass behave sensibly: shapes line up, bounds hold, the sum rule balances. This
file pins their actual numbers against the values in ``tests/fixtures/``,
recorded from the training implementation. It covers the augmented graph and
the predictions together, for both ``minmax_scaling`` settings and the demand
sum rule on and off.

All tests are offline: the fixture is read from disk, no model artifacts are
downloaded and no OPFDataset is instantiated.

Copyright 2026 UChicago Argonne, LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from lumina_inference.dataset.ingestion import load_from_dict
from lumina_inference.dataset.utils import hetero_batch_kwargs
from lumina_inference.dataset.virtual_node import variant_from_metadata
from lumina_inference.model.hetero_model import HGT

DATA_DIR = Path(__file__).parent / "fixtures"
TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def reference():
    """Load the training-reference fixture.

    Returns:
        dict: The parsed JSON metadata with a ``"tensors"`` key holding the
        companion safetensors payload, and ``"metadata"`` holding the
        ``(node_types, edge_types)`` pair in training's own order.
    """
    meta = json.loads((DATA_DIR / "training_reference.json").read_text())
    meta["tensors"] = load_file(DATA_DIR / "training_reference.safetensors")
    meta["metadata"] = (
        list(meta["node_types"]),
        [tuple(e.split("|")) for e in meta["edge_types"]],
    )
    meta["stats"] = tuple(tuple(h) for h in meta["demand_stats"])
    return meta


def _augment(reference):
    """Rebuild the augmented graph the way a served checkpoint would."""
    base = load_from_dict(copy.deepcopy(reference["opf_dict"]))
    variant = variant_from_metadata(reference["metadata"], reference["stats"])
    return variant(base)


def _model(reference, use_demand_sum_rule):
    """Build an HGT carrying the fixture's weights."""
    model = HGT(
        metadata=reference["metadata"],
        input_channels=reference["input_channels"],
        use_demand_sum_rule=use_demand_sum_rule,
        **reference["hyperparameters"],
    )
    weights = {
        k[len("weights."):]: v
        for k, v in reference["tensors"].items()
        if k.startswith("weights.")
    }
    model.load_state_dict(weights)
    return model.eval()


def _forward(reference, graph, use_demand_sum_rule, minmax_scaling):
    """Run the forward pass on ``graph`` exactly as the reference did."""
    model = _model(reference, use_demand_sum_rule)
    x = {nt: graph[nt].x.float() for nt in graph.node_types}
    edge_index = {et: graph[et].edge_index for et in graph.edge_types}
    edge_attr = {
        et: graph[et].edge_attr.float()
        for et in graph.edge_types
        if "edge_attr" in graph[et] and graph[et].edge_attr is not None
    }
    with torch.no_grad():
        return model(
            x,
            edge_index,
            edge_attr,
            minmax_scaling=minmax_scaling,
            **hetero_batch_kwargs(graph),
        )


class TestAugmentedGraphMatchesTraining:
    """The reconstructed graph must equal the one training built."""

    def test_the_fixture_describes_the_released_variant(self, reference):
        assert len(reference["node_types"]) == 5
        assert "grid" in reference["node_types"]
        assert len(reference["edge_types"]) == 18
        assert reference["input_channels"]["grid"] == 4

    def test_node_and_edge_types_match_in_order(self, reference):
        # HGTConv addresses relation weights by position in metadata[1], so the
        # order is part of the contract, not just the membership.
        graph = _augment(reference)
        assert list(graph.node_types) == reference["metadata"][0]
        assert list(graph.edge_types) == reference["metadata"][1]

    def test_node_features_match(self, reference):
        graph = _augment(reference)
        for node_type in reference["node_types"]:
            expected = reference["tensors"][f"graph.x.{node_type}"]
            assert torch.allclose(graph[node_type].x.float(), expected, **TOL), (
                f"{node_type}.x diverged from training"
            )

    def test_the_virtual_node_features_match(self, reference):
        # The standardised channels are the ones that depend on demand_stats,
        # so a wrong row or a wrong formula shows up here first.
        graph = _augment(reference)
        assert torch.allclose(
            graph["grid"].x.float(), reference["tensors"]["graph.x.grid"], **TOL
        )

    def test_star_edges_match(self, reference):
        graph = _augment(reference)
        for edge_type in reference["metadata"][1]:
            expected = reference["tensors"]["graph.edge_index." + "|".join(edge_type)]
            assert torch.equal(graph[edge_type].edge_index, expected), (
                f"{edge_type} edge_index diverged from training"
            )


class TestPredictionsMatchTraining:
    """The forward pass must reproduce training's outputs on shared weights."""

    @pytest.mark.parametrize("use_demand_sum_rule", [False, True])
    @pytest.mark.parametrize("minmax_scaling", [False, True])
    def test_predictions_match(self, reference, use_demand_sum_rule, minmax_scaling):
        graph = _augment(reference)
        out = _forward(reference, graph, use_demand_sum_rule, minmax_scaling)
        tag = f"sr{int(use_demand_sum_rule)}_mm{int(minmax_scaling)}"
        assert set(out) == {"bus", "generator"}
        for head, value in out.items():
            expected = reference["tensors"][f"pred.{tag}.{head}"]
            assert value.shape == expected.shape
            assert torch.allclose(value, expected, **TOL), (
                f"{head} predictions diverged from training for {tag}"
            )

    @pytest.mark.parametrize("use_demand_sum_rule", [False, True])
    @pytest.mark.parametrize("minmax_scaling", [False, True])
    def test_total_generation_matches(self, reference, use_demand_sum_rule, minmax_scaling):
        graph = _augment(reference)
        out = _forward(reference, graph, use_demand_sum_rule, minmax_scaling)
        tag = f"sr{int(use_demand_sum_rule)}_mm{int(minmax_scaling)}"
        expected = reference["expected_scalars"][f"sum_pg.{tag}"]
        assert out["generator"][:, 0].sum().item() == pytest.approx(expected, rel=1e-5)

    def test_the_sum_rule_decouples_total_generation_from_scaling(self, reference):
        # Training applies the sum rule independently of minmax_scaling, so the
        # two scaled/unscaled totals coincide only when the rule is on.
        scalars = reference["expected_scalars"]
        assert scalars["sum_pg.sr1_mm0"] == pytest.approx(scalars["sum_pg.sr1_mm1"], rel=1e-6)
        assert scalars["sum_pg.sr0_mm0"] != pytest.approx(scalars["sum_pg.sr0_mm1"], rel=1e-6)


class TestFixtureProvenance:
    """The fixture has to say what produced it, or it cannot be regenerated."""

    def test_provenance_is_recorded(self, reference):
        provenance = reference["provenance"]
        assert provenance["training_commit"]
        assert provenance["torch"]
        assert provenance["torch_geometric"]
