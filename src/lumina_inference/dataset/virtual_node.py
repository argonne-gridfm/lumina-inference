"""Virtual-node graph augmentation for LUMINA inference.

Some LUMINA checkpoints are trained on an augmented graph: one global
"grid" node wired bidirectionally to every other node type, plus reverse
counterparts for the transmission branch relations. Serving such a
checkpoint requires reproducing that augmentation on the input graph,
because the model's metadata declares node and edge types that plain OPF
ingestion does not produce.

The augmentation is data-only and deterministic — it reads ``load.x``,
``generator.x`` and the graph's type lists, never the solution targets —
so it is reproducible at inference time from the input graph alone, with
one exception: the standardised demand channels need per-case statistics,
which describe a pool of samples rather than any single graph. A checkpoint
carries those under ``config['virtual_node']['demand_stats']``, keyed by
case name; see :func:`select_demand_stats` to read that table and
:func:`compute_demand_stats` to build one.

:func:`variant_from_metadata` reconstructs the right variant directly
from a checkpoint's metadata, which is how :class:`~lumina_inference.modeler.Modeler`
decides whether a graph needs augmenting.

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

from __future__ import annotations

import math
from typing import Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

# Relations representing real transmission branches, which therefore need a
# reverse counterpart. The *_link relations (generator/load/shunt to bus) are
# already emitted in both directions by process_opf_dict, so reversing those
# would create duplicates.
BRANCH_RELATIONS = ("ac_line", "transformer")

# Node types of the OPF heterogeneous graph.
BASE_NODE_TYPES = ["bus", "generator", "load", "shunt"]
BASE_BRANCH_EDGE_TYPES = [("bus", "ac_line", "bus"), ("bus", "transformer", "bus")]

VARIANTS = ("legacy", "allnode_ones", "allnode_demand")


class AddReverseGridEdges(BaseTransform):
    """Add reverse edges for transmission branches as separate relations.

    For each branch relation ``('bus', rel, 'bus')`` present in the graph, adds
    ``('bus', f'{rel}_rev', 'bus')`` carrying the flipped ``edge_index`` and a copy
    of ``edge_attr`` (if any). The reverse edges live in their own relation so that
    physics code keyed on the original relation name keeps seeing each physical
    branch exactly once.

    Args:
        relations: Base relation names to reverse. Defaults to ac_line and
            transformer.
    """

    def __init__(self, relations: tuple[str, ...] = BRANCH_RELATIONS):
        self.relations = tuple(relations)

    def forward(self, data: HeteroData) -> HeteroData:
        """Add the reverse branch relations to ``data`` and return it.

        Note:
            Calling the transform (``transform(data)``) routes through
            ``BaseTransform.__call__``, which shallow-copies first, so the
            caller's graph is left alone and the result is the return value.
            Only a direct ``forward()`` call mutates the graph it is given.
        """
        for rel in self.relations:
            key = ("bus", rel, "bus")
            if key not in data.edge_types:
                continue
            rev_key = ("bus", f"{rel}_rev", "bus")
            data[rev_key].edge_index = data[key].edge_index.flip(0)
            edge_attr = getattr(data[key], "edge_attr", None)
            if edge_attr is not None:
                # Line impedance is reciprocal, so the reverse edge carries the same
                # physical parameters. HGT ignores edge_attr, but copying keeps the
                # graph well-formed for any other consumer.
                data[rev_key].edge_attr = edge_attr.clone()
        return data

    def reverse_edge_types(self, base_edge_types) -> list:
        """Return the edge types this transform would add to ``base_edge_types``.

        Lets the metadata builder stay in sync with the transform without
        duplicating the naming rule.

        Args:
            base_edge_types: Iterable of ``(src, rel, dst)`` tuples.

        Returns:
            List of new ``(src, rel, dst)`` tuples, in deterministic order.
        """
        present = {
            rel for (src, rel, dst) in base_edge_types
            if src == "bus" and dst == "bus"
        }
        return [
            ("bus", f"{rel}_rev", "bus")
            for rel in self.relations if rel in present
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(relations={self.relations})"


class AddHeteroVirtualNode(BaseTransform):
    """Add one global virtual node wired bidirectionally to chosen node types.

    Creates a single node of type ``virtual_node_type`` and, for every target node
    type ``nt``, the relation pair ``(nt, f'to_{virtual_node_type}', virtual_node_type)``
    and ``(virtual_node_type, f'from_{virtual_node_type}', nt)``.

    Args:
        virtual_node_type: Node-type name for the virtual node.
        connect_node_types: Node types to connect. ``None`` means every node type
            present in the sample (excluding the virtual node itself).
        init_feature: ``"global_demand"`` injects the global demand aggregates
            described below; ``"ones"`` emits a constant feature;
            ``"zeros"`` matches PyG's homogeneous ``VirtualNode`` convention.
        demand_stats: Optional ``((p_mean, p_std), (q_mean, q_std))`` used to
            standardise the two raw aggregates. Total demand carries a large
            constant offset relative to its spread, so without centring the input
            Linear would have to discover a near-exact bias cancellation before
            the signal became usable. A checkpoint records these; build your own
            over a pool of samples with :func:`compute_demand_stats`. If
            ``None``, the standardised channels are emitted as zeros, which keeps
            the feature width fixed so a model built for one setting still loads
            in the other.

    Feature layout for ``init_feature="global_demand"`` (width 4):
        0. ``(sum_Pd - mean) / std``  -- standardised total active demand
        1. ``(sum_Qd - mean) / std``  -- standardised total reactive demand
        2. ``sum_Pd / sum_Pmax``      -- system load factor, dimensionless
        3. ``sum_Qd / sum_Pmax``      -- reactive counterpart

    Channels 0-1 carry the within-case sample-to-sample variation the model has to
    track; channels 2-3 are self-normalising absolute quantities needing no external
    statistics, so they still mean something on an unseen case.
    """

    GLOBAL_DEMAND_DIM = 4

    def __init__(
        self,
        virtual_node_type: str = "grid",
        connect_node_types: Optional[list] = None,
        init_feature: str = "global_demand",
        demand_stats=None,
    ):
        if init_feature not in ("global_demand", "ones", "zeros"):
            raise ValueError(
                f"init_feature must be 'global_demand', 'ones' or 'zeros', "
                f"got {init_feature!r}"
            )
        self.virtual_node_type = virtual_node_type
        self.connect_node_types = (
            list(connect_node_types) if connect_node_types else None
        )
        self.init_feature = init_feature
        self.demand_stats = demand_stats

    @property
    def feature_dim(self) -> int:
        """Width of the virtual node's feature vector (its ``input_channels`` entry)."""
        return self.GLOBAL_DEMAND_DIM if self.init_feature == "global_demand" else 1

    def _build_feature(self, data: HeteroData) -> torch.Tensor:
        """Build the virtual node's ``(1, feature_dim)`` feature tensor."""
        if self.init_feature == "ones":
            return torch.ones(1, 1)
        if self.init_feature == "zeros":
            return torch.zeros(1, 1)

        sum_pd = data["load"].x[:, 0].sum()   # load column 0 = pd
        sum_qd = data["load"].x[:, 1].sum()   # load column 1 = qd
        # Total installed active capacity: a per-case constant, so dividing by it
        # yields a dimensionless load factor comparable across cases.
        sum_pmax = data["generator"].x[:, 3].sum().clamp(min=1e-6)  # column 3 = pmax

        if self.demand_stats is not None:
            (p_mean, p_std), (q_mean, q_std) = self.demand_stats
            z_p = (sum_pd - p_mean) / max(p_std, 1e-9)
            z_q = (sum_qd - q_mean) / max(q_std, 1e-9)
        else:
            z_p = torch.zeros((), dtype=sum_pd.dtype)
            z_q = torch.zeros((), dtype=sum_qd.dtype)

        return torch.stack(
            [
                torch.as_tensor(z_p, dtype=torch.float32),
                torch.as_tensor(z_q, dtype=torch.float32),
                (sum_pd / sum_pmax).float(),
                (sum_qd / sum_pmax).float(),
            ]
        ).view(1, self.GLOBAL_DEMAND_DIM)

    def forward(self, data: HeteroData) -> HeteroData:
        """Add the virtual node and its bidirectional star edges, returning ``data``.

        Note:
            As with :class:`AddReverseGridEdges`, calling the transform
            shallow-copies the graph first, so use the return value.
        """
        targets = self.connect_node_types or [
            nt for nt in data.node_types if nt != self.virtual_node_type
        ]

        data[self.virtual_node_type].x = self._build_feature(data)

        for nt in targets:
            if nt == self.virtual_node_type or nt not in data.node_types:
                continue
            num_nodes = data[nt].num_nodes
            if not num_nodes:
                continue
            idx = torch.arange(num_nodes, dtype=torch.long)
            zeros = torch.zeros(num_nodes, dtype=torch.long)
            data[
                (nt, f"to_{self.virtual_node_type}", self.virtual_node_type)
            ].edge_index = torch.stack([idx, zeros], dim=0)
            data[
                (self.virtual_node_type, f"from_{self.virtual_node_type}", nt)
            ].edge_index = torch.stack([zeros, idx], dim=0)
        return data

    def virtual_edge_types(self, base_node_types) -> list:
        """Return the edge types this transform would add for ``base_node_types``.

        Args:
            base_node_types: Node-type names present before the transform runs.

        Returns:
            List of new ``(src, rel, dst)`` tuples, in deterministic order.
        """
        targets = self.connect_node_types or [
            nt for nt in base_node_types if nt != self.virtual_node_type
        ]
        v = self.virtual_node_type
        out = []
        for nt in targets:
            out.append((nt, f"to_{v}", v))
            out.append((v, f"from_{v}", nt))
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"virtual_node_type={self.virtual_node_type!r}, "
            f"connect_node_types={self.connect_node_types}, "
            f"init_feature={self.init_feature!r}, "
            f"standardised={self.demand_stats is not None})"
        )


class Variant:
    """A named bundle of transform, extra relations, and virtual-node width.

    Keeping the three together is what stops the data and the model from
    drifting apart: a transform that adds relations the model's metadata does not
    list produces a silently mis-wired model rather than an error.

    Args:
        name: Variant name.
        reverse: The reverse-edge transform to apply first.
        virtual: The virtual-node transform to apply second.

    Attributes:
        name (str): See Args.
        extra_edge_types (list): Relations to append to the base metadata.
        virtual_node_dim (int): Input feature width of the virtual node.
        virtual_node_type (str): Node type name of the virtual node.
    """

    def __init__(
        self,
        name: str,
        reverse: AddReverseGridEdges,
        virtual: AddHeteroVirtualNode,
    ):
        self.name = name
        self._reverse = reverse
        self._virtual = virtual
        self.extra_edge_types = reverse.reverse_edge_types(
            BASE_BRANCH_EDGE_TYPES
        ) + virtual.virtual_edge_types(BASE_NODE_TYPES)
        self.virtual_node_dim = virtual.feature_dim
        self.virtual_node_type = virtual.virtual_node_type

    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply the reverse-edge and virtual-node transforms in order.

        Args:
            data: The graph to augment. Not modified — each sub-transform
                shallow-copies, so the augmented graph is the return value.

        Returns:
            HeteroData: The augmented graph.
        """
        return self._virtual(self._reverse(data))

    # Kept as an alias so the variant is usable wherever a bare callable transform
    # is expected as well as where ``.transform`` is looked up by name.
    transform = __call__

    def __repr__(self) -> str:
        return (
            f"Variant({self.name!r}, {len(self.extra_edge_types)} extra relations, "
            f"virtual_node_dim={self.virtual_node_dim}, {self._virtual!r})"
        )


def make_variant(name: str, demand_stats=None) -> Variant:
    """Build a named variant.

    Args:
        name: One of :data:`VARIANTS`.
        demand_stats: ``((p_mean, p_std), (q_mean, q_std))`` for the case being
            loaded. Required by ``allnode_demand`` — without it the standardised
            channels are constant zero, so the virtual node no longer carries the
            demand signal the weights were trained to read.

    Returns:
        Variant: The configured variant.

    Raises:
        ValueError: If ``name`` is unknown, or ``allnode_demand`` is requested
            without ``demand_stats``.
    """
    reverse = AddReverseGridEdges()

    if name == "legacy":
        virtual = AddHeteroVirtualNode(
            connect_node_types=["bus"], init_feature="ones"
        )
    elif name == "allnode_ones":
        virtual = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES, init_feature="ones"
        )
    elif name == "allnode_demand":
        if demand_stats is None:
            raise ValueError(
                "allnode_demand requires demand_stats (see compute_demand_stats); "
                "without them the standardised channels are constant and carry no "
                "signal"
            )
        virtual = AddHeteroVirtualNode(
            connect_node_types=BASE_NODE_TYPES,
            init_feature="global_demand",
            demand_stats=demand_stats,
        )
    else:
        raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")

    return Variant(name, reverse, virtual)


def extend_metadata(metadata: dict, variant: Variant) -> dict:
    """Extend a metadata dict with the variant's node and relations.

    Metadata is ``{"nodes": {name: in_dim}, "edges": {(s, r, d): attr_dim}}``
    describing the base OPF graph, so a transform that adds node or edge types
    would otherwise never reach the model.

    Args:
        metadata: A metadata dict. Not modified.
        variant: The variant whose transform will be applied to every sample.

    Returns:
        dict: A new metadata dict including the virtual node and the extra
        relations. Existing entries are left untouched, so this is idempotent.
    """
    nodes = dict(metadata["nodes"])
    edges = dict(metadata["edges"])

    nodes.setdefault(variant.virtual_node_type, variant.virtual_node_dim)

    for et in variant.extra_edge_types:
        et = tuple(et)
        if et in edges:
            continue
        src, rel, dst = et
        # Reverse branch relations carry a copy of the forward relation's edge_attr;
        # the virtual node's star edges carry none.
        base = (src, rel[: -len("_rev")], dst) if rel.endswith("_rev") else None
        edges[et] = edges.get(base, 0) if base else 0

    return {"nodes": nodes, "edges": edges}


def variant_from_metadata(metadata, demand_stats=None) -> Optional[Variant]:
    """Reconstruct the augmentation a checkpoint was trained with, from its metadata.

    A checkpoint's metadata fully determines the variant: the presence and width
    of the virtual node, which node types carry star edges, and which branch
    relations were reversed. Only ``demand_stats`` cannot be recovered, because
    it describes a pool of samples rather than any single graph.

    Args:
        metadata: The model's metadata, either ``{'nodes': ..., 'edges': ...}``
            or a ``(node_types, edge_types)`` pair. Edge types must be tuples.
        demand_stats: ``((p_mean, p_std), (q_mean, q_std))`` for the case being
            served. Required when the metadata declares a width-4 virtual node.

    Returns:
        Optional[Variant]: The variant to apply to each input graph, or ``None``
        when the metadata declares no virtual node (an unaugmented checkpoint,
        for which no transform is needed).

    Raises:
        ValueError: If the metadata declares a width-4 virtual node but no
            ``demand_stats`` is supplied, or if the reconstructed virtual node
            would not match the width the metadata declares.
    """
    if isinstance(metadata, dict):
        nodes, edge_types = metadata["nodes"], list(metadata["edges"])
    else:
        nodes, edge_types = metadata[0], list(metadata[1])
        # A (node_types, edge_types) pair carries no widths, so the virtual
        # node's declared width is unavailable; fall back to the demand layout
        # when stats are supplied and the constant feature otherwise.
        nodes = {
            nt: (AddHeteroVirtualNode.GLOBAL_DEMAND_DIM if demand_stats else 1)
            for nt in nodes
        }

    virtual_node_type = "grid"
    if virtual_node_type not in nodes:
        return None

    # Preserve metadata order: HGTConv addresses relation weights by position.
    connect_node_types = [
        src for (src, rel, dst) in edge_types
        if rel == f"to_{virtual_node_type}" and dst == virtual_node_type
    ]
    if not connect_node_types:
        raise ValueError(
            f"metadata declares a {virtual_node_type!r} node but no "
            f"'to_{virtual_node_type}' relations, so the virtual node would be "
            "isolated; the metadata is inconsistent"
        )

    width = int(nodes[virtual_node_type])
    if width == AddHeteroVirtualNode.GLOBAL_DEMAND_DIM:
        if demand_stats is None:
            raise ValueError(
                f"this checkpoint expects a width-{width} {virtual_node_type!r} "
                "node carrying standardised demand, so it needs demand_stats "
                "for the case being served: pass "
                "demand_stats=((p_mean, p_std), (q_mean, q_std)), e.g. from "
                "compute_demand_stats(samples). Without them the standardised "
                "channels would be constant zero, so the virtual node would not "
                "carry the demand signal the weights expect."
            )
        init_feature = "global_demand"
    else:
        init_feature = "ones"
        demand_stats = None

    reverse_relations = tuple(
        rel[: -len("_rev")] for (src, rel, dst) in edge_types
        if rel.endswith("_rev")
    )

    virtual = AddHeteroVirtualNode(
        virtual_node_type=virtual_node_type,
        connect_node_types=connect_node_types,
        init_feature=init_feature,
        demand_stats=demand_stats,
    )
    if virtual.feature_dim != width:
        raise ValueError(
            f"reconstructed {virtual_node_type!r} node is width "
            f"{virtual.feature_dim}, but the metadata declares {width}"
        )

    name = (
        "allnode_demand" if init_feature == "global_demand"
        else ("legacy" if connect_node_types == ["bus"] else "allnode_ones")
    )
    return Variant(name, AddReverseGridEdges(reverse_relations), virtual)


def needs_demand_stats(metadata) -> bool:
    """Whether a checkpoint's metadata declares a demand-carrying virtual node.

    Used to decide whether *failing* to resolve demand statistics is fatal. A
    width-1 virtual node carries a constant and needs none; a width-4 one
    standardises total demand and cannot be served without them.

    Args:
        metadata: The model's metadata, either ``{'nodes': ..., 'edges': ...}``
            or a ``(node_types, edge_types)`` pair.

    Returns:
        bool: True when the metadata declares a width-4 virtual node. The
        ``(node_types, edge_types)`` pair form carries no widths, so any
        virtual node at all counts -- deliberately conservative, since the
        caller uses this only to turn a failed lookup into an error.
    """
    if isinstance(metadata, dict):
        nodes = metadata.get("nodes") or {}
        return int(nodes.get("grid", 0)) == AddHeteroVirtualNode.GLOBAL_DEMAND_DIM
    return "grid" in (metadata[0] or ())


def select_demand_stats(table, case_name: Optional[str] = None):
    """Pick one case's demand statistics out of a per-case table.

    Total demand scales with the size of the network, so the statistics are
    per case and looked up by name. Applying another case's row leaves the
    standardised channels carrying no usable signal, which is why an ambiguous
    lookup is an error rather than a guess.

    Args:
        table: Mapping of case name to ``[[p_mean, p_std], [q_mean, q_std]]``,
            as recorded under ``config['virtual_node']['demand_stats']``. JSON
            nests these as lists; tuples are accepted equally.
        case_name: Which case to serve. Optional when the table holds exactly
            one entry.

    Returns:
        Optional[tuple]: ``((p_mean, p_std), (q_mean, q_std))`` as plain
        floats, or ``None`` when the table is empty.

    Raises:
        ValueError: If ``case_name`` is absent from a non-empty table, or if it
            is omitted for a table holding more than one case.
    """
    table = dict(table or {})
    if not table:
        return None

    if case_name is None:
        if len(table) > 1:
            raise ValueError(
                f"this checkpoint carries demand statistics for "
                f"{len(table)} cases, so which to apply is ambiguous: pass "
                f"case_name=<one of {sorted(table)}>. Total demand scales with "
                "the network, so another case's row leaves the standardised "
                "channels carrying no usable signal."
            )
        case_name = next(iter(table))
    elif case_name not in table:
        raise ValueError(
            f"no demand statistics for case {case_name!r}; the checkpoint "
            f"records {sorted(table) or '(none)'}. Pass demand_stats to supply "
            "your own, e.g. from compute_demand_stats(samples)."
        )

    (p_mean, p_std), (q_mean, q_std) = table[case_name]
    return ((float(p_mean), float(p_std)), (float(q_mean), float(q_std)))


def compute_demand_stats(samples):
    """Compute mean/std of total active and reactive demand over given samples.

    Args:
        samples: Iterable of ``HeteroData`` (or of objects exposing ``["load"].x``).

    Returns:
        tuple: ``((p_mean, p_std), (q_mean, q_std))`` as plain floats. The
        standard deviations are always finite and strictly positive.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    pd_sums, qd_sums = [], []
    for data in samples:
        pd_sums.append(float(data["load"].x[:, 0].sum()))
        qd_sums.append(float(data["load"].x[:, 1].sum()))
    if not pd_sums:
        raise ValueError("compute_demand_stats needs at least one sample")

    def _std(values) -> float:
        """Floor the deviation at 1e-9, NaN included.

        ``torch.std`` uses Bessel's correction, so a single sample yields NaN
        rather than 0.0 -- and ``max(nan, 1e-9)`` returns *nan*, because the
        comparison is false and Python keeps the first argument. Test explicitly
        for finiteness rather than relying on ``max``.
        """
        s = float(torch.tensor(values).std())
        return 1e-9 if not math.isfinite(s) or s < 1e-9 else s

    return (
        (float(torch.tensor(pd_sums).mean()), _std(pd_sums)),
        (float(torch.tensor(qd_sums).mean()), _std(qd_sums)),
    )
