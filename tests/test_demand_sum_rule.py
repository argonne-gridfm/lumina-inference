"""Tests for the demand sum rule in HGT.forward.

The sum rule ties each graph's total active generation to its total demand
plus an estimate of series losses, instead of letting every generator pick a
setpoint in isolation. These tests check the physics it promises, both of its
batching paths, and that it stays inert when switched off.

All tests are offline: no model artifacts are downloaded and no OPFDataset
is instantiated.

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

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from lumina_inference.dataset.utils import hetero_batch_kwargs
from lumina_inference.model.hetero_model import (
    _MAX_LOSS_FRACTION,
    _MIN_LOSS_FRACTION,
    HGT,
)

NUM_BUSES = 6
NUM_GENERATORS = 3
NUM_LOADS = 4

METADATA = {
    "nodes": {"bus": 7, "generator": 11, "load": 2, "shunt": 2},
    "edges": {
        ("bus", "ac_line", "bus"): 9,
        ("bus", "transformer", "bus"): 11,
        ("generator", "generator_link", "bus"): 0,
        ("bus", "generator_link", "generator"): 0,
        ("load", "load_link", "bus"): 0,
        ("bus", "load_link", "load"): 0,
        ("shunt", "shunt_link", "bus"): 0,
        ("bus", "shunt_link", "shunt"): 0,
    },
}
INPUT_CHANNELS = {"bus": 7, "generator": 11, "load": 2, "shunt": 2}


def _make_sample(seed: int) -> HeteroData:
    """Build one OPF-shaped graph with realistic bounds and impedances."""
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()

    bus = torch.rand(NUM_BUSES, 7, generator=g)
    bus[:, 1] = 0.94  # vmin
    bus[:, 2] = 1.06  # vmax
    data["bus"].x = bus

    gen = torch.rand(NUM_GENERATORS, 11, generator=g)
    gen[:, 2] = 0.1   # pmin
    gen[:, 3] = 3.0   # pmax
    gen[:, 5] = -1.0  # qmin
    gen[:, 6] = 1.0   # qmax
    data["generator"].x = gen

    # Demand comfortably inside aggregate [pmin, pmax] so the sum rule has a
    # feasible target rather than a clamped one.
    data["load"].x = torch.rand(NUM_LOADS, 2, generator=g) * 0.5 + 0.3
    data["shunt"].x = torch.rand(1, 2, generator=g)

    data["bus", "ac_line", "bus"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
    )
    ac_attr = torch.rand(5, 9, generator=g) * 0.05
    ac_attr[:, 4] = 0.02  # br_r
    ac_attr[:, 5] = 0.08  # br_x
    data["bus", "ac_line", "bus"].edge_attr = ac_attr

    data["bus", "transformer", "bus"].edge_index = torch.tensor([[0], [5]])
    xf_attr = torch.rand(1, 11, generator=g) * 0.05
    xf_attr[:, 2] = 0.01  # br_r
    xf_attr[:, 3] = 0.06  # br_x
    xf_attr[:, 7] = 1.02  # tap
    data["bus", "transformer", "bus"].edge_attr = xf_attr

    for key, edge_index in (
        (("generator", "generator_link", "bus"), [[0, 1, 2], [0, 1, 2]]),
        (("bus", "generator_link", "generator"), [[0, 1, 2], [0, 1, 2]]),
        (("load", "load_link", "bus"), [[0, 1, 2, 3], [1, 2, 3, 4]]),
        (("bus", "load_link", "load"), [[1, 2, 3, 4], [0, 1, 2, 3]]),
        (("shunt", "shunt_link", "bus"), [[0], [0]]),
        (("bus", "shunt_link", "shunt"), [[0], [0]]),
    ):
        data[key].edge_index = torch.tensor(edge_index)

    return data


def _make_model(use_demand_sum_rule: bool, seed: int = 0) -> HGT:
    """Build a small deterministic HGT."""
    torch.manual_seed(seed)
    model = HGT(
        metadata=METADATA,
        input_channels=INPUT_CHANNELS,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        num_heads=1,
        use_demand_sum_rule=use_demand_sum_rule,
    )
    model.eval()
    return model


def _predict(model, data, minmax_scaling=True):
    """Forward pass with the batch bookkeeping the sum rule needs."""
    with torch.no_grad():
        return model(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
            minmax_scaling=minmax_scaling,
            **hetero_batch_kwargs(data),
        )


@pytest.fixture
def single_graph():
    return _make_sample(seed=11)


@pytest.fixture
def batched_graphs():
    return Batch.from_data_list([_make_sample(11), _make_sample(22)])


# ---------------------------------------------------------------------------
# Default-off behaviour
# ---------------------------------------------------------------------------


class TestDefaultOff:
    """The sum rule must be inert unless explicitly enabled."""

    def test_defaults_to_disabled(self):
        assert _make_model(use_demand_sum_rule=False).use_demand_sum_rule is False

    def test_constructor_flag_is_coerced_to_bool(self):
        assert _make_model(use_demand_sum_rule=1).use_demand_sum_rule is True

    def test_off_leaves_outputs_bit_identical(self, batched_graphs):
        """Restructuring forward must not perturb the existing head."""
        off = _make_model(use_demand_sum_rule=False, seed=5)
        out = _predict(off, batched_graphs)

        # Same weights, reached through the pre-sum-rule code path shape.
        with torch.no_grad():
            expected = off(
                batched_graphs.x_dict,
                batched_graphs.edge_index_dict,
                batched_graphs.edge_attr_dict,
                minmax_scaling=True,
            )

        for key in ("bus", "generator"):
            assert torch.equal(out[key], expected[key])

    def test_off_needs_neither_edge_attrs_nor_batch_kwargs(self, single_graph):
        """A non-sum-rule model keeps working with the minimal call signature."""
        off = _make_model(use_demand_sum_rule=False)
        with torch.no_grad():
            out = off(single_graph.x_dict, single_graph.edge_index_dict)
        assert out["bus"].shape == (NUM_BUSES, 2)
        assert out["generator"].shape == (NUM_GENERATORS, 2)


# ---------------------------------------------------------------------------
# The physics the sum rule promises
# ---------------------------------------------------------------------------


class TestSumRulePhysics:
    """Generation must match demand plus a plausible loss fraction."""

    @pytest.mark.parametrize("minmax_scaling", [True, False])
    def test_generation_matches_demand_plus_losses(
        self, batched_graphs, minmax_scaling
    ):
        """Per graph: sum(pg) == demand * (1 + f), f inside the guard band."""
        model = _make_model(use_demand_sum_rule=True)
        out = _predict(model, batched_graphs, minmax_scaling=minmax_scaling)

        kwargs = hetero_batch_kwargs(batched_graphs)
        gen_batch = kwargs["batch_dict"]["generator"]
        load_batch = kwargs["batch_dict"]["load"]
        pg = out["generator"][:, 0]

        for graph in range(kwargs["num_graphs"]):
            pg_total = pg[gen_batch == graph].sum()
            demand = batched_graphs["load"].x[load_batch == graph][:, 0].sum()
            fraction = (pg_total / demand - 1.0).item()
            assert _MIN_LOSS_FRACTION - 1e-6 <= fraction <= _MAX_LOSS_FRACTION + 1e-6

    def test_generation_respects_bounds(self, batched_graphs):
        """Every setpoint stays inside its generator's [pmin, pmax]."""
        model = _make_model(use_demand_sum_rule=True)
        out = _predict(model, batched_graphs)

        pg = out["generator"][:, 0]
        pmin = batched_graphs["generator"].x[:, 2]
        pmax = batched_graphs["generator"].x[:, 3]
        assert (pg >= pmin - 1e-5).all()
        assert (pg <= pmax + 1e-5).all()

    def test_each_graph_is_solved_independently(self, batched_graphs):
        """A batched graph gets the same pg as it does predicted alone."""
        model = _make_model(use_demand_sum_rule=True)
        batched = _predict(model, batched_graphs)["generator"]
        alone = _predict(model, _make_sample(11))["generator"]

        assert torch.allclose(batched[:NUM_GENERATORS], alone, atol=1e-5)

    def test_applies_without_minmax_scaling(self, single_graph):
        """The sum rule is independent of minmax_scaling, so pg still shifts."""
        model = _make_model(use_demand_sum_rule=True)
        with_rule = _predict(model, single_graph, minmax_scaling=False)

        plain = _make_model(use_demand_sum_rule=False)
        plain.load_state_dict(model.state_dict())
        plain.eval()
        without_rule = _predict(plain, single_graph, minmax_scaling=False)

        assert not torch.allclose(
            with_rule["generator"][:, 0], without_rule["generator"][:, 0]
        )
        # qg, va and vm must be untouched by the sum rule.
        assert torch.equal(
            with_rule["generator"][:, 1], without_rule["generator"][:, 1]
        )
        assert torch.equal(with_rule["bus"], without_rule["bus"])

    def test_outputs_are_finite(self, batched_graphs):
        model = _make_model(use_demand_sum_rule=True)
        out = _predict(model, batched_graphs)
        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), f"non-finite output['{key}']"

    def test_single_unbatched_graph_uses_fallback_bookkeeping(self, single_graph):
        """hetero_batch_kwargs treats a bare HeteroData as one graph."""
        kwargs = hetero_batch_kwargs(single_graph)
        assert kwargs["num_graphs"] == 1
        assert kwargs["batch_dict"]["bus"].tolist() == [0] * NUM_BUSES

        out = _predict(_make_model(use_demand_sum_rule=True), single_graph)
        assert out["generator"].shape == (NUM_GENERATORS, 2)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSumRuleRequirements:
    """The sum rule needs branch parameters and batch bookkeeping."""

    def test_missing_edge_attrs_raises_value_error(self, single_graph):
        """Losses come from branch impedance, so edge_attr_dict is required."""
        model = _make_model(use_demand_sum_rule=True)
        with pytest.raises(ValueError, match="edge_attr_dict"):
            model(
                single_graph.x_dict,
                single_graph.edge_index_dict,
                None,
                minmax_scaling=True,
                **hetero_batch_kwargs(single_graph),
            )

    def test_missing_batch_kwargs_raises_key_error(self, single_graph):
        """Per-graph aggregation cannot proceed without batch_dict."""
        model = _make_model(use_demand_sum_rule=True)
        with pytest.raises(KeyError):
            model(
                single_graph.x_dict,
                single_graph.edge_index_dict,
                single_graph.edge_attr_dict,
                minmax_scaling=True,
            )

    def test_tolerates_a_missing_branch_relation(self, single_graph):
        """A grid with no transformers still gets a loss estimate."""
        del single_graph["bus", "transformer", "bus"]
        model = _make_model(use_demand_sum_rule=True)
        out = _predict(model, single_graph)
        assert torch.isfinite(out["generator"]).all()

    def test_demand_above_capacity_is_clamped_to_pmax(self, single_graph):
        """An infeasible target saturates at capacity instead of overshooting."""
        single_graph["load"].x[:, 0] = 100.0
        model = _make_model(use_demand_sum_rule=True)
        out = _predict(model, single_graph)

        pmax = single_graph["generator"].x[:, 3]
        assert (out["generator"][:, 0] <= pmax + 1e-5).all()
        assert torch.allclose(out["generator"][:, 0], pmax, atol=1e-3)
