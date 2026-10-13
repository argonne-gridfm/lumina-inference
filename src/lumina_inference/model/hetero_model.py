"""Heterogeneous Graph Transformer model for ACOPF inference.

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

import torch
import torch.nn.functional as F
from torch_geometric.nn import (HGTConv, Linear, global_add_pool,
                                global_max_pool)

_MIN_LOSS_FRACTION, _MAX_LOSS_FRACTION = 0.005, 0.06

_BRANCH_COLS = {
    ("bus", "ac_line", "bus"): {"br_r": 4, "br_x": 5},
    ("bus", "transformer", "bus"): {"br_r": 2, "br_x": 3, "tap": 7},
}


def _graph_totals(x, batch, n, col):
    """Per-graph sum of one node-feature column."""
    return global_add_pool(x[:, col].unsqueeze(-1), batch, n).squeeze(-1)


class HGT(torch.nn.Module):

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 dropout=0.0,
                 backend="sage",
                 edge_attr_dim=None,
                 use_demand_sum_rule=False,
                 **kwargs):
        r""" Heterogeneous Graph Transformer (HGT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
            use_demand_sum_rule (bool): Tie sum(Pg) to total demand instead of letting each
                generator pick alone. Applied independently of `minmax_scaling`, so qg, va
                and vm are untouched either way. Defaults to False.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.dropout = float(dropout)

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            self.node_types = list(metadata['nodes'].keys())
            self.edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple/list format: (node_types, edge_types)
            self.node_types = list(metadata[0])
            self.edge_types = [
                tuple(et) if isinstance(et, list) else et for et in metadata[1]
            ]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        for node_type in self.node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        metadata_tuple = (self.node_types, self.edge_types)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata_tuple, num_heads)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.use_demand_sum_rule = bool(use_demand_sum_rule)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=True, **kwargs):
        """Forward pass of the HGT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to True.

        Returns:
            dict: Output predictions for each target node type.
        """

        # physical values before input transform, needed for total demand
        _x_raw = x_dict

        if minmax_scaling or self.use_demand_sum_rule:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: torch.relu(self.lin_dict[node_type](x_dict[node_type]))
            for node_type in self.node_types
        }
        if self.dropout > 0.0:
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            if self.dropout > 0.0:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if not (minmax_scaling or self.use_demand_sum_rule):
            return {"bus": bus_out, "generator": gen_out}

        # Create new tensors instead of modifying in-place
        bus_out_final = bus_out.clone()
        gen_out_final = gen_out.clone()

        if minmax_scaling:
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

        if self.use_demand_sum_rule:
            # Independent of minmax_scaling: some checkpoints are trained without
            # it, and turning it on only to reach this head shifts qg, va and vm.
            loss_fraction = self._estimate_loss_fraction(
                bus_out_final, edge_index_dict, edge_attr_dict, _x_raw, kwargs)
            gen_out_final[:, 0] = self._sum_rule_pg(
                gen_out[:, 0], _pmin, _pmax, _x_raw, loss_fraction, kwargs)

        return {"bus": bus_out_final, "generator": gen_out_final}

    def _estimate_loss_fraction(self, bus_out, edge_index_dict, edge_attr_dict,
                                _x_raw, kwargs):
        """Series losses as a fraction of demand, per graph, from the predicted voltages.

        For a branch i-j with series admittance g + jb and tap t, the active loss is
        g * (|v_i/t|^2 + |v_j|^2 - 2 |v_i/t||v_j| cos(va_i - va_j)).
        """
        if edge_attr_dict is None:
            raise ValueError(
                "the sum rule derives transmission losses from branch parameters, so "
                "forward needs edge_attr_dict; pass batch.edge_attr_dict")
        batch_dict, num_graphs = kwargs["batch_dict"], kwargs["num_graphs"]
        bus_batch = batch_dict["bus"]
        va, vm = bus_out[:, 0], bus_out[:, 1]
        pd_tot = _graph_totals(_x_raw["load"], batch_dict["load"], num_graphs,
                               0)  # load column 0 = pd

        total = torch.zeros_like(pd_tot)
        for rel, cols in _BRANCH_COLS.items():
            ei, ea = edge_index_dict.get(rel), edge_attr_dict.get(rel)
            if ei is None or ea is None or not ei.numel():
                continue
            i, j = ei[0], ei[1]
            r, x = ea[:, cols["br_r"]], ea[:, cols["br_x"]]
            g = r / (r ** 2 + x ** 2).clamp_min(1e-12)
            tap = ea[:, cols["tap"]].abs().clamp_min(1e-6) if "tap" in cols else 1.0
            vi, vj = vm[i] / tap, vm[j]
            per_edge = g * (vi ** 2 + vj ** 2 - 2 * vi * vj * torch.cos(va[i] - va[j]))
            total = total + global_add_pool(
                per_edge.unsqueeze(-1), bus_batch[i], num_graphs).squeeze(-1)

        return (total / pd_tot.clamp_min(1e-6)).clamp(_MIN_LOSS_FRACTION, _MAX_LOSS_FRACTION)

    def _sum_rule_pg(self, zz, _pmin, _pmax, _x_raw, loss_fraction, kwargs):
        """
        Solves one scaling factor per graph to capture aggregate demand like:

            sum_i (pmin_i + span_i * sigmoid(z_i + lam)) = demand * (1 + f)

        instead of letting each generator pick its own setpoint in isolation.
        Returns the per-generator active power.
        """
        _PD_COL = 0    # load column 0 = pd
        _PMIN_COL = 2  # generator column 2 = pmin

        batch_dict, num_graphs = kwargs["batch_dict"], kwargs["num_graphs"]
        gb = batch_dict["generator"]
        span = (_pmax - _pmin).clamp_min(0.0)

        pd_tot = _graph_totals(_x_raw["load"], batch_dict["load"], num_graphs, _PD_COL)
        pmin_tot = _graph_totals(_x_raw["generator"], gb, num_graphs, _PMIN_COL)

        # Phi(lam) = sum_i span_i * sigmoid(z_i + lam) is strictly increasing over
        # (0, sum span), so clamping the target into that range is what guarantees the
        # bisection brackets a root.
        span_tot = global_add_pool(span.unsqueeze(-1), gb, num_graphs).squeeze(-1)
        resid = (pd_tot * (1.0 + loss_fraction) - pmin_tot).clamp_min(0.0)
        resid = torch.minimum(resid, span_tot)

        # Bracket in closed form: at lam = -max(z) - 40 every sigmoid is ~0 and at
        # lam = -min(z) + 40 every sigmoid is ~1, so the root is enclosed whatever the
        # logits are. A fixed +-40 silently converges to an endpoint on real checkpoints.
        with torch.no_grad():
            zmax = global_max_pool(zz.unsqueeze(-1), gb, num_graphs).squeeze(-1)
            zmin = -global_max_pool((-zz).unsqueeze(-1), gb, num_graphs).squeeze(-1)
            lo, hi = -zmax - 40.0, -zmin + 40.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                tot = global_add_pool(
                    (span * torch.sigmoid(zz + mid[gb])).unsqueeze(-1),
                    gb, num_graphs).squeeze(-1)
                over = tot > resid
                hi = torch.where(over, mid, hi)
                lo = torch.where(over, lo, mid)
            lam0 = 0.5 * (lo + hi)

        # lam0 reaches autograd as a constant, this adds a no-op in forward (resid == f0)
        # and differentiates to exactly the implicit derivative
        # dlam/dz_j = -span_j sg_j (1 - sg_j) / sum_i span_i sg_i (1 - sg_i).

        sg0 = torch.sigmoid(zz + lam0[gb])
        f0 = global_add_pool((span * sg0).unsqueeze(-1), gb, num_graphs).squeeze(-1)
        slope = global_add_pool(
            (span * sg0 * (1 - sg0)).unsqueeze(-1), gb, num_graphs).squeeze(-1)
        lam = lam0 + (resid - f0) / slope.clamp_min(1e-9)

        return _pmin + span * torch.sigmoid(zz + lam[gb])
