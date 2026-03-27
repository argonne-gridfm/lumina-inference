"""Heterogeneous Graph Neural Network model for ACOPF inference.

Vendored from lumina-core model/opf/hetero_model — OPFHeteroGNN only.
Training-specific model variants (RGAT, HEAT, HGT, etc.) are excluded.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GINConv,
    GraphConv,
    HeteroConv,
    Linear,
    MLP,
    SAGEConv,
)


class OPFHeteroGNN(torch.nn.Module):
    """Heterogeneous Graph Neural Network (HeteroGNN) model for OPF.

    Supports both OPF-specific and generic heterogeneous graph inference.
    When ``output_node_types`` is not provided, defaults to the standard
    OPF output types (``bus`` and ``generator``, each with 2 outputs).

    Args:
        metadata (dict or tuple): Metadata containing node types and edge types.
            If dict: ``{'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}``
            If tuple: ``(node_types, edge_types)``
        input_channels (dict): Number of input features for each node type.
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample. Defaults to 2.
        num_layers (int): Number of layers. Defaults to 3.
        backend (str): Graph convolutional layer backend. Defaults to ``"sage"``.
        edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        output_node_types (dict, optional): Mapping of output node type names
            to their output dimensions. If None, defaults to
            ``{'bus': out_channels, 'generator': out_channels}`` for OPF
            backward compatibility.
        scaling_config (dict, optional): Configuration for output scaling.
            If None, OPF-specific minmax scaling is used when the data
            contains ``bus`` and ``generator`` node types. For generic data,
            no scaling is applied. Custom configs can specify per-node-type
            scaling rules.
    """

    # Default OPF output types for backward compatibility
    _DEFAULT_OPF_OUTPUT_TYPES = {"bus": 2, "generator": 2}

    def __init__(
        self,
        metadata,
        input_channels,
        hidden_channels=64,
        out_channels=2,
        num_layers=3,
        backend="sage",
        edge_attr_dim=None,
        output_node_types=None,
        scaling_config=None,
        **kwargs,
    ):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata["nodes"].keys())
            edge_types = list(metadata["edges"].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]
            edge_types = metadata[1]

        self._node_types = node_types
        self.backend = backend
        self.edge_attr_support = backend == "gat"
        self.scaling_config = scaling_config

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(
                    f"input_channels must contain entry for node type '{node_type}'"
                )
            self.lin_dict[node_type] = Linear(
                input_channels[node_type], hidden_channels
            )

        # Heterogeneous graph convolutional layers for edges
        self.convs = torch.nn.ModuleList()

        # Get edge attribute dimensions from metadata
        if isinstance(metadata, dict):
            edge_attr_dims = {
                edge_type: (dim if dim > 0 else None)
                for edge_type, dim in metadata["edges"].items()
            }
        else:
            # Fallback to hardcoded dimensions for legacy format
            edge_attr_dims = {
                ("bus", "ac_line", "bus"): 9,
                ("bus", "transformer", "bus"): 11,
                ("generator", "generator_link", "bus"): None,
                ("bus", "generator_link", "generator"): None,
                ("load", "load_link", "bus"): None,
                ("bus", "load_link", "load"): None,
                ("shunt", "shunt_link", "bus"): None,
                ("bus", "shunt_link", "shunt"): None,
            }

        def get_conv_layer(edge_type):
            if backend == "sage":
                return SAGEConv(
                    (hidden_channels, hidden_channels), hidden_channels
                )
            elif backend == "gcn":
                return GraphConv(
                    (hidden_channels, hidden_channels), hidden_channels
                )
            elif backend == "gin":
                return GINConv(MLP([hidden_channels, hidden_channels]))
            elif backend == "gat":
                edge_dim = edge_attr_dims.get(edge_type, None)
                return GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    edge_type: get_conv_layer(edge_type)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # Output layers for target node types
        # Use provided output_node_types or default to OPF types
        if output_node_types is not None:
            self._output_node_types = output_node_types
        else:
            self._output_node_types = {
                k: out_channels for k in self._DEFAULT_OPF_OUTPUT_TYPES
            }

        self.out_dict = torch.nn.ModuleDict(
            {
                node_type: Linear(hidden_channels, dim)
                for node_type, dim in self._output_node_types.items()
            }
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            for rel_conv in conv.convs.values():
                if hasattr(rel_conv, "reset_parameters"):
                    rel_conv.reset_parameters()
            conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def _is_opf_schema(self, x_dict):
        """Check if the input data follows the OPF schema.

        Returns True if both ``bus`` and ``generator`` are present in
        x_dict with sufficient feature dimensions for OPF scaling.
        """
        return (
            "bus" in x_dict
            and "generator" in x_dict
            and x_dict["bus"].shape[-1] >= 3
            and x_dict["generator"].shape[-1] >= 7
        )

    def _extract_bounds_opf(self, x_dict):
        """Extract OPF-specific bounds from input features.

        Returns:
            dict: Bounds for bus voltage and generator power limits.
        """
        return {
            "vmin": x_dict["bus"][:, 1].clone(),
            "vmax": x_dict["bus"][:, 2].clone(),
            "pmin": x_dict["generator"][:, 2].clone(),
            "pmax": x_dict["generator"][:, 3].clone(),
            "qmin": x_dict["generator"][:, 5].clone(),
            "qmax": x_dict["generator"][:, 6].clone(),
        }

    def _apply_scaling_opf(self, outputs, bounds):
        """Apply OPF-specific minmax scaling to outputs.

        Scales bus voltage magnitude (column 1) and generator active/reactive
        power (columns 0, 1) using sigmoid activation bounded by the
        extracted limits.

        Args:
            outputs (dict): Raw output tensors keyed by node type.
            bounds (dict): OPF bounds from :meth:`_extract_bounds_opf`.

        Returns:
            dict: Scaled output tensors.
        """
        result = {}

        if "bus" in outputs:
            bus_out = outputs["bus"]
            bus_out_final = bus_out.clone()
            bus_out_final[:, 1] = (
                F.sigmoid(bus_out[:, 1])
                * (bounds["vmax"] - bounds["vmin"])
                + bounds["vmin"]
            )
            result["bus"] = bus_out_final

        if "generator" in outputs:
            gen_out = outputs["generator"]
            gen_out_final = gen_out.clone()
            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = (
                gen_out_sigmoid[:, 0]
                * (bounds["pmax"] - bounds["pmin"])
                + bounds["pmin"]
            )
            gen_out_final[:, 1] = (
                gen_out_sigmoid[:, 1]
                * (bounds["qmax"] - bounds["qmin"])
                + bounds["qmin"]
            )
            result["generator"] = gen_out_final

        # Pass through any other output types unchanged
        for key in outputs:
            if key not in result:
                result[key] = outputs[key]

        return result

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict=None,
        minmax_scaling=False,
        **kwargs,
    ):
        """Forward pass of the HeteroGNN model.

        Supports both OPF-specific and generic heterogeneous graph data.
        When ``minmax_scaling`` is True:

        - **OPF data** (detected by presence of ``bus`` and ``generator``
          node types with sufficient features): applies OPF-specific
          sigmoid-bounded scaling using voltage and power limits from
          the input features.
        - **Generic data**: returns raw outputs without scaling (scaling
          can be applied externally via ``scaling_config``).

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type.
            minmax_scaling (bool): Whether to apply min-max scaling to outputs.

        Returns:
            dict: Output predictions for each target node type.
        """
        is_opf = self._is_opf_schema(x_dict)

        # Extract bounds before input transformation destroys them
        bounds = None
        if minmax_scaling and is_opf:
            bounds = self._extract_bounds_opf(x_dict)

        # Transform input features
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message passing
        for conv in self.convs:
            if self.edge_attr_support and edge_attr_dict is not None:
                x_dict = conv(
                    x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
                )
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {
                    key: F.dropout(x, p=0.1, training=self.training)
                    for key, x in x_dict.items()
                }
            else:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {
                    key: F.dropout(x, p=0.1, training=self.training)
                    for key, x in x_dict.items()
                }

        # Final predictions for all output node types
        outputs = {}
        for node_type in self._output_node_types:
            if node_type in x_dict:
                outputs[node_type] = self.out_dict[node_type](
                    x_dict[node_type]
                )

        # Apply scaling
        if minmax_scaling and is_opf and bounds is not None:
            return self._apply_scaling_opf(outputs, bounds)
        else:
            return outputs
