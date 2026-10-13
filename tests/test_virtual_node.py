"""Tests for virtual-node graph augmentation.

Checkpoints trained on an augmented graph declare node and edge types that
plain OPF ingestion does not produce, so serving them means reproducing the
augmentation exactly. These tests cover the transforms, the metadata
bookkeeping that keeps data and model in step, and the reconstruction of a
variant from a checkpoint's own metadata.

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
from torch_geometric.data import HeteroData

from lumina_inference.dataset.virtual_node import (
    BASE_NODE_TYPES,
    VARIANTS,
    AddHeteroVirtualNode,
    AddReverseGridEdges,
    compute_demand_stats,
    extend_metadata,
    make_variant,
    needs_demand_stats,
    select_demand_stats,
    variant_from_metadata,
)

NUM_BUSES = 5
NUM_GENERATORS = 2
NUM_LOADS = 3
NUM_SHUNTS = 1

BASE_METADATA = {
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

DEMAND_STATS = ((1.5, 0.1), (0.6, 0.05))


def _make_sample(pd_scale: float = 1.0) -> HeteroData:
    """Build a small OPF-shaped graph."""
    data = HeteroData()
    data["bus"].x = torch.ones(NUM_BUSES, 7)

    gen = torch.ones(NUM_GENERATORS, 11)
    gen[:, 3] = 2.0  # pmax, so sum_pmax == 4.0
    data["generator"].x = gen

    load = torch.ones(NUM_LOADS, 2)
    load[:, 0] = 0.5 * pd_scale  # pd, so sum_pd == 1.5 * pd_scale
    load[:, 1] = 0.2 * pd_scale  # qd, so sum_qd == 0.6 * pd_scale
    data["load"].x = load

    data["shunt"].x = torch.ones(NUM_SHUNTS, 2)

    data["bus", "ac_line", "bus"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]]
    )
    data["bus", "ac_line", "bus"].edge_attr = torch.arange(
        27, dtype=torch.float32
    ).view(3, 9)
    data["bus", "transformer", "bus"].edge_index = torch.tensor([[0], [4]])
    data["bus", "transformer", "bus"].edge_attr = torch.arange(
        11, dtype=torch.float32
    ).view(1, 11)

    for key, edge_index in (
        (("generator", "generator_link", "bus"), [[0, 1], [0, 1]]),
        (("bus", "generator_link", "generator"), [[0, 1], [0, 1]]),
        (("load", "load_link", "bus"), [[0, 1, 2], [1, 2, 3]]),
        (("bus", "load_link", "load"), [[1, 2, 3], [0, 1, 2]]),
        (("shunt", "shunt_link", "bus"), [[0], [0]]),
        (("bus", "shunt_link", "shunt"), [[0], [0]]),
    ):
        data[key].edge_index = torch.tensor(edge_index)

    return data


@pytest.fixture
def sample():
    return _make_sample()


# ---------------------------------------------------------------------------
# Reverse branch edges
# ---------------------------------------------------------------------------


class TestAddReverseGridEdges:
    """Branch relations get a reverse counterpart in their own relation."""

    def test_adds_reverse_relations(self, sample):
        sample = AddReverseGridEdges()(sample)
        assert ("bus", "ac_line_rev", "bus") in sample.edge_types
        assert ("bus", "transformer_rev", "bus") in sample.edge_types

    def test_reverse_edge_index_is_flipped(self, sample):
        forward = sample["bus", "ac_line", "bus"].edge_index.clone()
        sample = AddReverseGridEdges()(sample)
        reverse = sample["bus", "ac_line_rev", "bus"].edge_index
        assert torch.equal(reverse, forward.flip(0))

    def test_reverse_edge_attr_is_a_copy(self, sample):
        """Impedance is reciprocal, so attributes carry over -- but not aliased."""
        sample = AddReverseGridEdges()(sample)
        forward = sample["bus", "ac_line", "bus"]
        reverse = sample["bus", "ac_line_rev", "bus"]
        assert torch.equal(reverse.edge_attr, forward.edge_attr)
        assert reverse.edge_attr is not forward.edge_attr

    def test_forward_relation_is_left_intact(self, sample):
        """Physics code keyed on the base relation must see each branch once."""
        before = sample["bus", "ac_line", "bus"].edge_index.clone()
        sample = AddReverseGridEdges()(sample)
        assert torch.equal(sample["bus", "ac_line", "bus"].edge_index, before)

    def test_absent_relation_is_skipped(self, sample):
        del sample["bus", "transformer", "bus"]
        sample = AddReverseGridEdges()(sample)
        assert ("bus", "transformer_rev", "bus") not in sample.edge_types
        assert ("bus", "ac_line_rev", "bus") in sample.edge_types

    def test_link_relations_are_not_reversed(self, sample):
        """The *_link relations already exist in both directions."""
        sample = AddReverseGridEdges()(sample)
        assert not [
            et for et in sample.edge_types if et[1].endswith("_link_rev")
        ]

    def test_reverse_edge_types_matches_the_transform(self, sample):
        transform = AddReverseGridEdges()
        declared = transform.reverse_edge_types(
            [("bus", "ac_line", "bus"), ("bus", "transformer", "bus")]
        )
        before = set(sample.edge_types)
        sample = transform(sample)
        assert set(declared) == set(sample.edge_types) - before


# ---------------------------------------------------------------------------
# The virtual node itself
# ---------------------------------------------------------------------------


class TestAddHeteroVirtualNode:
    """One global node, wired bidirectionally to each target node type."""

    def test_rejects_unknown_init_feature(self):
        with pytest.raises(ValueError, match="init_feature"):
            AddHeteroVirtualNode(init_feature="bogus")

    def test_demand_feature_is_width_four(self, sample):
        sample = AddHeteroVirtualNode(demand_stats=DEMAND_STATS)(sample)
        assert sample["grid"].x.shape == (1, 4)

    def test_ones_feature_is_width_one(self, sample):
        sample = AddHeteroVirtualNode(init_feature="ones")(sample)
        assert sample["grid"].x.shape == (1, 1)
        assert sample["grid"].x.item() == 1.0

    def test_zeros_feature_is_width_one(self, sample):
        sample = AddHeteroVirtualNode(init_feature="zeros")(sample)
        assert sample["grid"].x.shape == (1, 1)
        assert sample["grid"].x.item() == 0.0

    def test_feature_dim_matches_emitted_width(self, sample):
        transform = AddHeteroVirtualNode(demand_stats=DEMAND_STATS)
        sample = transform(sample)
        assert transform.feature_dim == sample["grid"].x.shape[1]

    def test_load_factor_channels_are_self_normalising(self, sample):
        """Channels 2-3 are sum_Pd/sum_Pmax and sum_Qd/sum_Pmax."""
        sample = AddHeteroVirtualNode(demand_stats=DEMAND_STATS)(sample)
        feature = sample["grid"].x[0]
        # sum_pd == 1.5, sum_qd == 0.6, sum_pmax == 4.0
        assert feature[2].item() == pytest.approx(1.5 / 4.0)
        assert feature[3].item() == pytest.approx(0.6 / 4.0)

    def test_standardised_channels_use_demand_stats(self):
        """Channels 0-1 centre and scale the raw aggregates."""
        data = AddHeteroVirtualNode(demand_stats=DEMAND_STATS)(
            _make_sample(pd_scale=1.2)
        )
        feature = data["grid"].x[0]
        assert feature[0].item() == pytest.approx((1.5 * 1.2 - 1.5) / 0.1, abs=1e-5)
        assert feature[1].item() == pytest.approx((0.6 * 1.2 - 0.6) / 0.05, abs=1e-5)

    def test_standardised_channels_are_zero_without_stats(self, sample):
        """Width is preserved so a model built either way still loads."""
        sample = AddHeteroVirtualNode(demand_stats=None)(sample)
        feature = sample["grid"].x[0]
        assert sample["grid"].x.shape == (1, 4)
        assert feature[0].item() == 0.0
        assert feature[1].item() == 0.0
        # The self-normalising channels still carry signal.
        assert feature[2].item() == pytest.approx(1.5 / 4.0)

    def test_star_edges_exist_in_both_directions(self, sample):
        sample = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES, init_feature="ones"
        )(sample)
        for node_type in BASE_NODE_TYPES:
            assert (node_type, "to_grid", "grid") in sample.edge_types
            assert ("grid", "from_grid", node_type) in sample.edge_types

    def test_star_edge_indices_point_at_the_single_grid_node(self, sample):
        sample = AddHeteroVirtualNode(
            connect_node_types=["bus"], init_feature="ones"
        )(sample)
        to_grid = sample["bus", "to_grid", "grid"].edge_index
        from_grid = sample["grid", "from_grid", "bus"].edge_index

        assert to_grid.tolist() == [list(range(NUM_BUSES)), [0] * NUM_BUSES]
        assert from_grid.tolist() == [[0] * NUM_BUSES, list(range(NUM_BUSES))]

    def test_every_node_reaches_the_grid_node(self, sample):
        """The star is what makes the graph globally connected in two hops."""
        sample = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES, init_feature="ones"
        )(sample)
        for node_type in BASE_NODE_TYPES:
            senders = sample[node_type, "to_grid", "grid"].edge_index[0]
            assert sorted(senders.tolist()) == list(
                range(sample[node_type].num_nodes)
            )

    def test_star_edges_carry_no_attributes(self, sample):
        sample = AddHeteroVirtualNode(
            connect_node_types=["bus"], init_feature="ones"
        )(sample)
        store = sample["bus", "to_grid", "grid"]
        assert getattr(store, "edge_attr", None) is None

    def test_connect_node_types_none_connects_everything_present(self, sample):
        sample = AddHeteroVirtualNode(
            connect_node_types=None, init_feature="ones"
        )(sample)
        for node_type in BASE_NODE_TYPES:
            assert (node_type, "to_grid", "grid") in sample.edge_types

    def test_empty_node_type_is_skipped(self, sample):
        """A grid with no shunts must not get a zero-width star relation."""
        sample["shunt"].x = torch.zeros(0, 2)
        sample = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES, init_feature="ones"
        )(sample)
        assert ("shunt", "to_grid", "grid") not in sample.edge_types
        assert ("bus", "to_grid", "grid") in sample.edge_types

    def test_virtual_edge_types_matches_the_transform(self, sample):
        transform = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES, init_feature="ones"
        )
        declared = transform.virtual_edge_types(BASE_NODE_TYPES)
        before = set(sample.edge_types)
        sample = transform(sample)
        assert set(declared) == set(sample.edge_types) - before


# ---------------------------------------------------------------------------
# Named variants
# ---------------------------------------------------------------------------


class TestMakeVariant:
    """Variants bundle the transform with the metadata it implies."""

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="unknown variant"):
            make_variant("nonexistent")

    def test_allnode_demand_requires_stats(self):
        """Without stats the standardised channels silently carry no signal."""
        with pytest.raises(ValueError, match="demand_stats"):
            make_variant("allnode_demand")

    def test_legacy_connects_only_buses(self, sample):
        sample = make_variant("legacy")(sample)
        assert ("bus", "to_grid", "grid") in sample.edge_types
        assert ("generator", "to_grid", "grid") not in sample.edge_types
        assert sample["grid"].x.shape == (1, 1)

    def test_allnode_ones_connects_every_type(self, sample):
        sample = make_variant("allnode_ones")(sample)
        for node_type in BASE_NODE_TYPES:
            assert (node_type, "to_grid", "grid") in sample.edge_types
        assert sample["grid"].x.shape == (1, 1)

    def test_allnode_demand_emits_demand_features(self, sample):
        sample = make_variant("allnode_demand", demand_stats=DEMAND_STATS)(sample)
        assert sample["grid"].x.shape == (1, 4)

    @pytest.mark.parametrize("name", VARIANTS)
    def test_variant_applies_both_transforms(self, name):
        stats = DEMAND_STATS if name == "allnode_demand" else None
        data = make_variant(name, demand_stats=stats)(_make_sample())
        assert "grid" in data.node_types
        assert ("bus", "ac_line_rev", "bus") in data.edge_types

    def test_calling_a_variant_leaves_the_input_untouched(self):
        """Modeler.predict_single relies on this to not mutate the caller's graph."""
        data = _make_sample()
        before_nodes, before_edges = set(data.node_types), set(data.edge_types)

        augmented = make_variant("allnode_demand", demand_stats=DEMAND_STATS)(data)

        assert set(data.node_types) == before_nodes
        assert set(data.edge_types) == before_edges
        assert "grid" not in data.node_types
        assert "grid" in augmented.node_types

    def test_declared_extra_edge_types_cover_what_is_added(self):
        """A relation the metadata omits means a silently mis-wired model."""
        data = _make_sample()
        variant = make_variant("allnode_demand", demand_stats=DEMAND_STATS)
        before = set(data.edge_types)
        data = variant(data)
        assert set(data.edge_types) - before == set(variant.extra_edge_types)


# ---------------------------------------------------------------------------
# Metadata bookkeeping
# ---------------------------------------------------------------------------


class TestExtendMetadata:
    """Metadata must describe exactly the graph the transform produces."""

    def test_adds_the_virtual_node_width(self):
        variant = make_variant("allnode_demand", demand_stats=DEMAND_STATS)
        extended = extend_metadata(BASE_METADATA, variant)
        assert extended["nodes"]["grid"] == 4

    def test_ones_variant_declares_width_one(self):
        extended = extend_metadata(BASE_METADATA, make_variant("allnode_ones"))
        assert extended["nodes"]["grid"] == 1

    def test_does_not_mutate_the_input(self):
        before = dict(BASE_METADATA["nodes"])
        extend_metadata(BASE_METADATA, make_variant("allnode_ones"))
        assert BASE_METADATA["nodes"] == before
        assert "grid" not in BASE_METADATA["nodes"]

    def test_reverse_relations_inherit_the_forward_attr_width(self):
        extended = extend_metadata(BASE_METADATA, make_variant("allnode_ones"))
        assert extended["edges"][("bus", "ac_line_rev", "bus")] == 9
        assert extended["edges"][("bus", "transformer_rev", "bus")] == 11

    def test_star_relations_declare_no_attributes(self):
        extended = extend_metadata(BASE_METADATA, make_variant("allnode_ones"))
        assert extended["edges"][("bus", "to_grid", "grid")] == 0
        assert extended["edges"][("grid", "from_grid", "bus")] == 0

    def test_is_idempotent(self):
        variant = make_variant("allnode_demand", demand_stats=DEMAND_STATS)
        once = extend_metadata(BASE_METADATA, variant)
        twice = extend_metadata(once, variant)
        assert once == twice

    def test_base_entries_survive_unchanged(self):
        extended = extend_metadata(BASE_METADATA, make_variant("allnode_ones"))
        for key, value in BASE_METADATA["edges"].items():
            assert extended["edges"][key] == value


# ---------------------------------------------------------------------------
# Reconstructing a variant from a checkpoint's metadata
# ---------------------------------------------------------------------------


def _vn_metadata(variant_name, stats=None):
    """Metadata as a checkpoint trained with this variant would record it."""
    return extend_metadata(
        BASE_METADATA, make_variant(variant_name, demand_stats=stats)
    )


class TestVariantFromMetadata:
    """A checkpoint's metadata fully determines its augmentation."""

    def test_returns_none_without_a_virtual_node(self):
        assert variant_from_metadata(BASE_METADATA) is None

    def test_reconstructs_allnode_demand(self):
        metadata = _vn_metadata("allnode_demand", DEMAND_STATS)
        variant = variant_from_metadata(metadata, demand_stats=DEMAND_STATS)
        assert variant.name == "allnode_demand"
        assert variant.virtual_node_dim == 4

    def test_reconstructs_allnode_ones(self):
        variant = variant_from_metadata(_vn_metadata("allnode_ones"))
        assert variant.name == "allnode_ones"
        assert variant.virtual_node_dim == 1

    def test_reconstructs_legacy(self):
        variant = variant_from_metadata(_vn_metadata("legacy"))
        assert variant.name == "legacy"
        assert variant.virtual_node_dim == 1

    @pytest.mark.parametrize("name", VARIANTS)
    def test_round_trips_every_variant(self, name):
        """The reconstruction must produce the same graph as the original."""
        stats = DEMAND_STATS if name == "allnode_demand" else None
        metadata = _vn_metadata(name, stats)

        original = make_variant(name, demand_stats=stats)(_make_sample())
        reconstructed = variant_from_metadata(
            metadata, demand_stats=stats
        )(_make_sample())

        assert set(original.node_types) == set(reconstructed.node_types)
        assert set(original.edge_types) == set(reconstructed.edge_types)
        assert torch.equal(original["grid"].x, reconstructed["grid"].x)

    def test_reconstructed_graph_satisfies_the_metadata(self):
        """Every declared node type and width must be present after transform."""
        metadata = _vn_metadata("allnode_demand", DEMAND_STATS)
        data = variant_from_metadata(
            metadata, demand_stats=DEMAND_STATS
        )(_make_sample())

        for node_type, width in metadata["nodes"].items():
            assert node_type in data.node_types
            assert data[node_type].x.shape[1] == width

    def test_demand_node_without_stats_raises(self):
        """Hard error rather than silently zeroing the standardised channels."""
        metadata = _vn_metadata("allnode_demand", DEMAND_STATS)
        with pytest.raises(ValueError, match="demand_stats"):
            variant_from_metadata(metadata)

    def test_ones_variant_needs_no_stats(self):
        assert variant_from_metadata(_vn_metadata("allnode_ones")) is not None

    def test_stats_are_ignored_for_a_width_one_node(self):
        """A constant-feature checkpoint must not be handed demand features."""
        variant = variant_from_metadata(
            _vn_metadata("allnode_ones"), demand_stats=DEMAND_STATS
        )
        data = variant(_make_sample())
        assert data["grid"].x.shape == (1, 1)

    def test_isolated_virtual_node_is_rejected(self):
        """A grid node with no star relations means inconsistent metadata."""
        metadata = {
            "nodes": {**BASE_METADATA["nodes"], "grid": 1},
            "edges": dict(BASE_METADATA["edges"]),
        }
        with pytest.raises(ValueError, match="isolated"):
            variant_from_metadata(metadata)

    def test_preserves_metadata_edge_order(self):
        """HGTConv addresses relation weights by position, so order matters."""
        metadata = _vn_metadata("allnode_demand", DEMAND_STATS)
        variant = variant_from_metadata(metadata, demand_stats=DEMAND_STATS)

        declared = [
            et for et in metadata["edges"] if et[1] == "to_grid"
        ]
        reconstructed = [
            et for et in variant.extra_edge_types if et[1] == "to_grid"
        ]
        assert declared == reconstructed

    def test_accepts_a_node_and_edge_type_pair(self):
        """Legacy tuple-style metadata is still understood."""
        metadata = _vn_metadata("allnode_ones")
        pair = (list(metadata["nodes"]), list(metadata["edges"]))
        assert variant_from_metadata(pair) is not None


# ---------------------------------------------------------------------------
# Demand statistics
# ---------------------------------------------------------------------------


class TestComputeDemandStats:
    """Statistics for the standardised channels."""

    def test_rejects_an_empty_sample_set(self):
        with pytest.raises(ValueError, match="at least one sample"):
            compute_demand_stats([])

    def test_means_match_the_totals(self):
        samples = [_make_sample(1.0), _make_sample(3.0)]
        (p_mean, _), (q_mean, _) = compute_demand_stats(samples)
        assert p_mean == pytest.approx((1.5 + 4.5) / 2)
        assert q_mean == pytest.approx((0.6 + 1.8) / 2)

    def test_single_sample_std_is_finite_and_positive(self):
        """torch.std uses Bessel's correction, so one sample yields NaN."""
        (_, p_std), (_, q_std) = compute_demand_stats([_make_sample()])
        assert p_std > 0.0
        assert q_std > 0.0
        assert torch.isfinite(torch.tensor([p_std, q_std])).all()

    def test_identical_samples_give_a_floored_std(self):
        (_, p_std), _ = compute_demand_stats([_make_sample(), _make_sample()])
        assert p_std == pytest.approx(1e-9)

    def test_output_feeds_make_variant(self):
        stats = compute_demand_stats([_make_sample(1.0), _make_sample(2.0)])
        data = make_variant("allnode_demand", demand_stats=stats)(_make_sample())
        assert torch.isfinite(data["grid"].x).all()


# ---------------------------------------------------------------------------
# Reading the per-case table a checkpoint carries
# ---------------------------------------------------------------------------


# Two cases whose demand scales differ widely, which is why the choice of row
# cannot be guessed.
CASE_SMALL = "pglib_opf_case14_ieee"
CASE_LARGE = "pglib_opf_case13659_pegase"
STATS_TABLE = {
    CASE_SMALL: [[2.5, 0.125], [0.75, 0.035]],
    CASE_LARGE: [[3800.0, 11.4], [985.0, 4.5]],
}


class TestNeedsDemandStats:
    """Whether a failed statistics lookup should be fatal."""

    def test_width_four_grid_needs_them(self):
        assert needs_demand_stats({"nodes": {"bus": 7, "grid": 4}, "edges": {}})

    def test_width_one_grid_does_not(self):
        assert not needs_demand_stats({"nodes": {"bus": 7, "grid": 1}, "edges": {}})

    def test_absent_grid_does_not(self):
        assert not needs_demand_stats({"nodes": {"bus": 7}, "edges": {}})

    def test_pair_form_is_conservative(self):
        """The pair form carries no widths, so any grid node counts."""
        assert needs_demand_stats((["bus", "grid"], []))
        assert not needs_demand_stats((["bus"], []))

    def test_matches_a_variant_built_metadata(self):
        base = {"nodes": dict.fromkeys(BASE_NODE_TYPES, 2), "edges": {}}
        demand = extend_metadata(
            base, make_variant("allnode_demand", demand_stats=((1.0, 1.0), (1.0, 1.0)))
        )
        ones = extend_metadata(base, make_variant("allnode_ones"))
        assert needs_demand_stats(demand)
        assert not needs_demand_stats(ones)


class TestSelectDemandStats:
    """Picking one case's row out of the table."""

    def test_empty_table_yields_none(self):
        assert select_demand_stats({}) is None
        assert select_demand_stats(None) is None

    def test_named_case_is_returned(self):
        assert select_demand_stats(STATS_TABLE, CASE_SMALL) == (
            (2.5, 0.125),
            (0.75, 0.035),
        )

    def test_single_entry_needs_no_name(self):
        assert select_demand_stats({CASE_SMALL: STATS_TABLE[CASE_SMALL]}) == (
            (2.5, 0.125),
            (0.75, 0.035),
        )

    def test_several_entries_without_a_name_is_an_error(self):
        with pytest.raises(ValueError, match="case_name"):
            select_demand_stats(STATS_TABLE)

    def test_the_error_lists_the_candidates(self):
        with pytest.raises(ValueError, match=CASE_LARGE):
            select_demand_stats(STATS_TABLE)

    def test_unknown_case_lists_what_is_recorded(self):
        with pytest.raises(ValueError, match="pglib_opf_case30_ieee"):
            select_demand_stats(STATS_TABLE, "pglib_opf_case30_ieee")
        with pytest.raises(ValueError, match=CASE_SMALL):
            select_demand_stats(STATS_TABLE, "pglib_opf_case30_ieee")

    def test_nested_lists_are_coerced_to_floats(self):
        """JSON produces lists of ints and floats, never tuples."""
        stats = select_demand_stats({CASE_SMALL: [[1, 2], [3, 4]]}, CASE_SMALL)
        assert stats == ((1.0, 2.0), (3.0, 4.0))
        assert all(isinstance(n, float) for half in stats for n in half)

    def test_tuples_are_accepted_equally(self):
        assert select_demand_stats({CASE_SMALL: ((1.0, 2.0), (3.0, 4.0))}) == (
            (1.0, 2.0),
            (3.0, 4.0),
        )

    def test_the_caller_is_not_handed_the_table(self):
        """A returned row must not alias the config the caller still holds."""
        table = {CASE_SMALL: [[1.0, 2.0], [3.0, 4.0]]}
        select_demand_stats(table, CASE_SMALL)
        assert table == {CASE_SMALL: [[1.0, 2.0], [3.0, 4.0]]}

    def test_output_feeds_make_variant(self):
        stats = select_demand_stats(STATS_TABLE, CASE_SMALL)
        data = make_variant("allnode_demand", demand_stats=stats)(_make_sample())
        assert torch.isfinite(data["grid"].x).all()
