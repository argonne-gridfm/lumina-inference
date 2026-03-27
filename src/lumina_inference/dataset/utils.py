"""Utility functions for processing OPF datasets.

Vendored from lumina-core dataset/opf/utils — edge index helpers only.

Copyright (c) 2026, Argonne National Laboratory
All rights reserved.
"""

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    """Extract edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Edge index tensor.
    """
    return torch.tensor(
        np.array(
            [
                obj["grid"]["edges"][edge_name]["senders"],
                obj["grid"]["edges"][edge_name]["receivers"],
            ]
        )
    )


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    """Extract reversed edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Reversed edge index tensor.
    """
    return torch.tensor(
        np.array(
            [
                obj["grid"]["edges"][edge_name]["receivers"],
                obj["grid"]["edges"][edge_name]["senders"],
            ]
        )
    )
