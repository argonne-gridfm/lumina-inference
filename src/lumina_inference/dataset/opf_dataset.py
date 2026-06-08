"""Customized OPFDataset class for the ACOPF task (inference only).

Vendored from lumina-core dataset/opf/opf_dataset.
Supports both JSON and HDF5 data formats.

Copyright 2026 UChicago Argonne, LLC.
All rights reserved.
"""

import json
import os
import os.path as osp
import pickle
import shutil
import stat
import warnings
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_tar

from lumina_inference.dataset.ingestion import process_opf_dict
from lumina_inference.dataset.utils import extract_edge_index, extract_edge_index_rev


class OPFDataset(InMemoryDataset):
    r"""The heterogeneous OPF data from the `"Large-scale Datasets for AC
    Optimal Power Flow with Topological Perturbations"
    <https://arxiv.org/abs/2406.07234>`_ paper.

    :class:`OPFDataset` is a large-scale dataset of solved optimal power flow
    problems, derived from the
    `pglib-opf <https://github.com/power-grid-lib/pglib-opf>`_ dataset.

    The physical topology of the grid is represented by the :obj:`"bus"` node
    type, and the connecting AC lines and transformers. Additionally,
    :obj:`"generator"`, :obj:`"load"`, and :obj:`"shunt"` nodes are connected
    to :obj:`"bus"` nodes using a dedicated edge type each, *e.g.*,
    :obj:`"generator_link"`.

    Edge direction corresponds to the properties of the line, *e.g.*,
    :obj:`b_fr` is the line charging susceptance at the :obj:`from`
    (source/sender) bus.

    Args:
        root (str): Root directory where the dataset should be saved.
        case_name (str, optional): The name of the original pglib-opf case.
            (default: :obj:`"pglib_opf_case14_ieee"`)
        group_id (int, optional): The specific group to load. Each group
            contains 15,000 samples. Valid values are [0, 19].
            (default: :obj:`0`)
        topological_perturbations (bool, optional): Whether to use the dataset
            with added topological perturbations. (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes
            in a :obj:`torch_geometric.data.HeteroData` object and returns
            a transformed version. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a boolean
            value. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        keep_temp (bool, optional): Whether to keep the temporary files
            after processing. (default: :obj:`False`)
        n_jobs (int, optional): The number of jobs to use for parallel
            processing. If set to :obj:`-1`, all available cores will be used.
            NOTE: for larger dataset, it is recommended to set this to a lower
            positive value to avoid memory issues. (default: :obj:`-1`)
        local_raw_folder (str, optional): Local folder to look for raw files.
            (default: :obj:`None`)
    """

    url = "https://storage.googleapis.com/gridopt-dataset"

    def __init__(
        self,
        root: str,
        case_name: Literal[
            "pglib_opf_case14_ieee",
            "pglib_opf_case30_ieee",
            "pglib_opf_case57_ieee",
            "pglib_opf_case118_ieee",
            "pglib_opf_case500_goc",
            "pglib_opf_case2000_goc",
            "pglib_opf_case4661_sdet",
            "pglib_opf_case6470_rte",
            "pglib_opf_case10000_goc",
            "pglib_opf_case13659_pegase",
        ] = "pglib_opf_case14_ieee",
        group_id: int = 0,
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        keep_temp: bool = False,
        n_jobs: int = -1,
        local_raw_folder: str = None,
    ) -> None:
        self.case_name = case_name
        self.group_id = group_id
        self.topological_perturbations = topological_perturbations

        self._raw_root = osp.join(root, "OPFData/raw")
        self._processed_root = osp.join(root, "OPFData/processed")
        self._release = "dataset_release_1"
        if topological_perturbations:
            self._release += "_nminusone"
        self.n_jobs = n_jobs
        self.keep_temp = keep_temp

        self.local_raw_folder = local_raw_folder
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )

        # Load only the specified group
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

        if osp.exists(self._processed_root):
            try:
                current_mode = os.stat(self._processed_root).st_mode
                os.chmod(self._processed_root, current_mode | stat.S_IWGRP)
            except OSError as exc:
                warnings.warn(
                    f"Failed to set group write permission on "
                    f"{self._processed_root}: {exc}"
                )

    @property
    def raw_dir(self) -> str:
        """Raw data folder."""
        return osp.join(self._raw_root, self._release)

    @property
    def processed_dir(self) -> str:
        """Processed data folder."""
        return osp.join(
            self._processed_root, self._release, self.case_name
        )

    @property
    def tmp_dir(self) -> str:
        """Temporary data folder."""
        return osp.join(
            self.raw_dir,
            "gridopt-dataset-tmp",
            self._release,
            self.case_name,
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names, which are stored locally."""
        return [f"{self.case_name}_{self.group_id}.tar.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names, which are stored locally."""
        return [f"group_{self.group_id}.pt"]

    def download(self) -> None:
        """Download .tar.gz files."""
        print("Download files")
        self.download_and_extract(self.raw_file_names[0])
        print(f"Downloaded {self.raw_file_names[0]} to {self.raw_dir}")

    def download_and_extract(self, name: str) -> None:
        """Download and extract a .tar.gz file."""
        url = f"{self.url}/{self._release}/{name}"
        path = download_url(url, self.raw_dir)
        extract_tar(path, self.raw_dir)

    def process(self) -> None:
        """Process the raw files into a single file.

        Automatically detects HDF5 vs JSON format and dispatches
        to the appropriate processing method.
        """
        h5_files = [f for f in self.raw_paths if f.endswith(".h5")]

        if h5_files:
            print(f"HDF5 files detected: {h5_files}")
            try:
                self.process_hdf5_group(self.group_id)
            except Exception as e:
                print(f"Error processing HDF5 group {self.group_id}: {e}")
                raise e
            return

        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        try:
            self.process_json_group(self.group_id)
        except Exception as e:
            print(f"Error processing group {self.group_id}: {e}")
            raise e

        print(f"Processed group {self.group_id}")

        # Remove tmp_dir content to save local space
        if not self.keep_temp:
            shutil.rmtree(osp.join(self.raw_dir, "gridopt-dataset-tmp"))

    def _post_process_and_save(
        self,
        data_list: List[Optional[Union[HeteroData, List[HeteroData]]]],
        group_id: int,
    ):
        """Helper to filter, transform, collate and save processed data."""
        flattened_list = []
        for item in data_list:
            if item is None:
                continue
            if isinstance(item, list):
                flattened_list.extend(item)
            else:
                flattened_list.append(item)

        data_list = flattened_list

        if self.pre_filter is not None or self.pre_transform is not None:
            if self.pre_filter is not None:
                data_list = [
                    data for data in data_list if self.pre_filter(data)
                ]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save(
            (self._data, self.slices),
            osp.join(self.processed_dir, f"group_{group_id}.pt"),
        )

    def process_json_group(self, group_id: int):
        """Process a single group of JSON files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """
        group_json_files = glob(
            osp.join(self.tmp_dir, f"group_{group_id}", "*.json")
        )

        if len(group_json_files) < 15000:
            extract_tar(
                osp.join(self.raw_dir, self.raw_file_names[0]), self.raw_dir
            )
            group_json_files = glob(
                osp.join(self.tmp_dir, f"group_{group_id}", "*.json")
            )

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(process_json_file)(fn)
            for fn in tqdm.tqdm(group_json_files, desc=f"Group {group_id}")
        )

        self._post_process_and_save(data_list, group_id)

    def process_hdf5_group(self, group_id: int):
        """Process a single group of HDF5 files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """
        import h5py as _h5py

        raw_paths = self.raw_paths
        h5_files = [f for f in raw_paths if f.endswith(".h5")]

        if not h5_files:
            print(f"No HDF5 files found in {self.raw_dir}")
            return

        tasks = []
        for h5_file in h5_files:
            with _h5py.File(h5_file, "r") as f:
                for scenario_key in f.keys():
                    tasks.append((h5_file, scenario_key))

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_process_hdf5_scenario_from_path)(fn, key)
            for fn, key in tqdm.tqdm(tasks, desc=f"Group {group_id} HDF5")
        )

        self._post_process_and_save(data_list, group_id)

    def metadata(self):
        """Returns the metadata of the dataset."""
        return {
            "nodes": {
                "bus": self._data["bus"].x.size(1),
                "generator": self._data["generator"].x.size(1),
                "load": self._data["load"].x.size(1),
                "shunt": self._data["shunt"].x.size(1),
            },
            "edges": {
                ("bus", "ac_line", "bus"): self._data[
                    "bus", "ac_line", "bus"
                ].edge_attr.size(1),
                ("bus", "transformer", "bus"): self._data[
                    "bus", "transformer", "bus"
                ].edge_attr.size(1),
                ("generator", "generator_link", "bus"): 0,
                ("bus", "generator_link", "generator"): 0,
                ("load", "load_link", "bus"): 0,
                ("bus", "load_link", "load"): 0,
                ("shunt", "shunt_link", "bus"): 0,
                ("bus", "shunt_link", "shunt"): 0,
            },
        }

    def __repr__(self) -> str:
        """Returns the string representation of the dataset."""
        return (
            f"{self.__class__.__name__}({len(self)}, "
            f"case_name={self.case_name}, "
            f"topological_perturbations={self.topological_perturbations})"
        )


# ---------------------------------------------------------------------------
# JSON processing
# ---------------------------------------------------------------------------


def process_json_file(json_file, require_solution=True):
    """Process a single JSON file into a HeteroData object.

    This function delegates to :func:`~lumina_inference.dataset.ingestion.process_opf_dict`
    for the actual conversion. It can be used both for training (with solution
    data) and for inference (without solution data).

    Args:
        json_file (str): Path to the JSON file.
        require_solution (bool): If True (default), the JSON must contain
            a ``'solution'`` key. Set to False for inference-only usage
            where ground-truth solutions are not available.

    Returns:
        HeteroData or None: Processed data object, or None if JSON
        decoding fails.
    """
    with open(json_file) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {json_file}")
            return None

    return process_opf_dict(obj, require_solution=require_solution)


# ---------------------------------------------------------------------------
# HDF5 processing
# ---------------------------------------------------------------------------


def _lazy_import_h5py():
    """Lazy import of h5py to avoid hard dependency."""
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install it with: pip install lumina-inference[hdf5]"
        ) from e


def _lazy_import_schemas():
    """Lazy import of schema classes to avoid hard pydantic dependency."""
    from lumina_inference.dataset.schema import (
        ContingencyH5Load,
        ContingencyH5LoadSolution,
        H5ACLine,
        H5Bus,
        H5BusSolution,
        H5EdgeSolution,
        H5Generator,
        H5GeneratorSolution,
        H5Load,
        H5Shunt,
        H5Transformer,
        JSONACLine,
        JSONBus,
        JSONBusSolution,
        JSONEdgeSolution,
        JSONGenerator,
        JSONGeneratorSolution,
        JSONLoad,
        JSONShunt,
        JSONTransformer,
    )
    return {
        "ContingencyH5Load": ContingencyH5Load,
        "ContingencyH5LoadSolution": ContingencyH5LoadSolution,
        "H5ACLine": H5ACLine,
        "H5Bus": H5Bus,
        "H5BusSolution": H5BusSolution,
        "H5EdgeSolution": H5EdgeSolution,
        "H5Generator": H5Generator,
        "H5GeneratorSolution": H5GeneratorSolution,
        "H5Load": H5Load,
        "H5Shunt": H5Shunt,
        "H5Transformer": H5Transformer,
        "JSONACLine": JSONACLine,
        "JSONBus": JSONBus,
        "JSONBusSolution": JSONBusSolution,
        "JSONEdgeSolution": JSONEdgeSolution,
        "JSONGenerator": JSONGenerator,
        "JSONGeneratorSolution": JSONGeneratorSolution,
        "JSONLoad": JSONLoad,
        "JSONShunt": JSONShunt,
        "JSONTransformer": JSONTransformer,
    }


# Module-level cache for schema classes
_SCHEMAS = None


def _get_schemas():
    """Get schema classes, importing lazily on first use."""
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = _lazy_import_schemas()
    return _SCHEMAS


def process_hdf5_scenario(
    scenario, scenario_key: str
) -> Union[Optional[HeteroData], List[HeteroData]]:
    """Process a single scenario from HDF5 format to HeteroData.

    Args:
        scenario: An h5py Group representing a single scenario.
        scenario_key (str): The key/name of the scenario in the HDF5 file.

    Returns:
        HeteroData, List[HeteroData], or None: Processed data object(s).
            Returns a list for contingency scenarios, a single HeteroData
            for standard ACOPF scenarios, or None on error.
    """
    try:
        # contingency scenario or acopf scenario
        if "base_solution" in scenario:
            return process_contingency_scenario(scenario, scenario_key)

        grid = scenario["grid"]
        solution = scenario["solution"]
        metadata = scenario["metadata"]

        hdata = HeteroData()
        _process_nodes_hdf5(hdata, grid, solution)
        _process_edges_hdf5(hdata, grid, solution)

        hdata.baseMVA = torch.tensor(
            grid["context"]["baseMVA"][()], dtype=torch.float32
        ).view(-1)
        hdata.objective = torch.tensor(
            metadata.attrs["objective"], dtype=torch.float32
        )
        hdata.scenario_id = scenario_key

        return hdata
    except Exception as e:
        print(f"Error in process_hdf5_scenario for {scenario_key}: {e}")
        return None


def process_contingency_scenario(
    scenario, scenario_key: str
) -> List[HeteroData]:
    """Process a contingency scenario from HDF5 format.

    Contingency scenarios contain a base solution and multiple
    post-contingency solutions. Each converged post-contingency
    solution is returned as a separate HeteroData object.

    Args:
        scenario: An h5py Group representing a contingency scenario.
        scenario_key (str): The key/name of the scenario.

    Returns:
        List[HeteroData]: One HeteroData per converged post-contingency
        solution. Empty list on error.
    """
    h5py = _lazy_import_h5py()

    try:
        if "grid" in scenario:
            grid = scenario["grid"]
        elif "grid" in scenario.file:
            grid = scenario.file["grid"]
        elif "grid" in scenario.parent:
            grid = scenario.parent["grid"]
        else:
            grid = scenario["base_solution"]["grid"]

        try:
            base_mva = torch.tensor(
                grid["context"]["baseMVA"][()], dtype=torch.float32
            ).view(-1)
        except Exception:
            base_mva = torch.tensor([100.0], dtype=torch.float32)

        metadata = scenario["metadata"]
        n_contingencies = int(metadata.attrs.get("n_contingencies", 0))

        results = []

        if "post_contingency" in scenario:
            pc_group = scenario["post_contingency"]
            for cont_name in pc_group.keys():
                cont_group = pc_group[cont_name]

                if "opf" not in cont_group:
                    continue

                solution = cont_group["opf"]
                if solution.attrs.get("opf_converged", 1) != 1:
                    continue

                hdata = HeteroData()
                _process_nodes_hdf5(hdata, grid, solution)
                _process_edges_hdf5(hdata, grid, solution)

                hdata.baseMVA = base_mva
                hdata.objective = torch.tensor(
                    solution.attrs["objective"], dtype=torch.float32
                )
                hdata.scenario_id = f"{scenario_key}_{cont_name}"
                hdata.n_contingencies = n_contingencies

                results.append(hdata)

        return results
    except Exception as e:
        print(
            f"Error in process_contingency_scenario for {scenario_key}: {e}"
        )
        return []


def _process_hdf5_scenario_from_path(h5_file_path, scenario_key):
    """Process a single HDF5 scenario by opening the file and reading the key.

    Used for parallel processing where each worker opens its own file handle.
    """
    h5py = _lazy_import_h5py()
    with h5py.File(h5_file_path, "r") as f:
        scenario = f[scenario_key]
        return process_hdf5_scenario(scenario, scenario_key)


def _align_features(data: np.ndarray, src_schema, dst_schema) -> np.ndarray:
    """Align feature columns between HDF5 and JSON schemas.

    Feature indices are different between HDF5 ACOPF, JSON ACOPF,
    and HDF5 contingency formats and must be aligned.

    Args:
        data: Input feature array of shape ``[num_items, num_src_features]``.
        src_schema: Source schema class (e.g. ``H5Bus``).
        dst_schema: Destination schema class (e.g. ``JSONBus``).

    Returns:
        Aligned feature array of shape ``[num_items, num_dst_features]``.
    """
    mapping = src_schema.get_alignment_map(dst_schema)
    aligned = np.zeros((data.shape[0], len(dst_schema.get_feature_names())))
    for src_idx, dst_idx in mapping.items():
        aligned[:, dst_idx] = data[:, src_idx]
    return aligned


def _process_nodes_hdf5(hdata: HeteroData, grid, solution):
    """Process node data from HDF5 format aligned with JSON schema."""
    S = _get_schemas()
    nodes = grid["nodes"]
    sol_nodes = solution["nodes"]

    required_fields = ["bus", "load", "generator"]
    missing_fields = [f for f in required_fields if f not in nodes]

    if missing_fields:
        raise Exception(
            f"Invalid HDF5 file: missing required fields: {missing_fields}"
        )

    # --- bus ---
    bus_data = nodes["bus"][()]
    if bus_data.shape[0] == 5:
        bus_data = bus_data.T

    if bus_data.shape[1] >= 5:
        # HDF5 bus: vmin, vmax, zone, area, bus_type
        # JSON bus: base_kv, bus_type, vmin, vmax
        aligned_bus = _align_features(bus_data, S["H5Bus"], S["JSONBus"])

        # augment hdf5 data with base_kv if missing
        json_indices = S["JSONBus"].get_field_indices()
        if "base_kv" in json_indices and np.all(
            aligned_bus[:, json_indices["base_kv"]] == 0
        ):
            aligned_bus[:, json_indices["base_kv"]] = 1.0

        # One-hot encode bus_type (JSON loader does this; HDF5 uses category)
        bus_type = aligned_bus[:, json_indices["bus_type"]].astype(int)
        bus_type_onehot = np.eye(4)[bus_type - 1]
        bus_x_wo_type = np.delete(
            aligned_bus, json_indices["bus_type"], axis=1
        )
        bus_x_final = np.concatenate([bus_x_wo_type, bus_type_onehot], axis=1)
    else:
        bus_x_final = bus_data

    hdata["bus"].x = torch.tensor(bus_x_final, dtype=torch.float32)

    # --- generator ---
    gen_data = nodes["generator"][()]
    if gen_data.shape[0] == 10:
        gen_data = gen_data.T

    if gen_data.shape[1] >= 10:
        gen_x_final = _align_features(
            gen_data, S["H5Generator"], S["JSONGenerator"]
        )
    else:
        gen_x_final = gen_data
    hdata["generator"].x = torch.tensor(gen_x_final, dtype=torch.float32)

    # --- load ---
    load_data = nodes["load"][()]
    if load_data.shape[0] == 4 or load_data.shape[0] == 2:
        load_data = load_data.T

    if load_data.shape[1] == 4:
        load_x_final = _align_features(
            load_data, S["ContingencyH5Load"], S["JSONLoad"]
        )
    elif load_data.shape[1] == 2:
        load_x_final = _align_features(load_data, S["H5Load"], S["JSONLoad"])
    else:
        load_x_final = load_data

    hdata["load"].x = torch.tensor(load_x_final, dtype=torch.float32)

    # --- shunt ---
    if "shunt" in nodes:
        shunt_data = nodes["shunt"][()]
        if shunt_data.shape[0] == 2:
            shunt_data = shunt_data.T

        if shunt_data.shape[1] == 2:
            shunt_x_final = _align_features(
                shunt_data, S["H5Shunt"], S["JSONShunt"]
            )
        else:
            shunt_x_final = shunt_data
        hdata["shunt"].x = torch.tensor(shunt_x_final, dtype=torch.float32)

    # --- solution data ---
    if "bus" in sol_nodes:
        bus_sol = sol_nodes["bus"][()]
        if bus_sol.shape[0] == 2:
            bus_sol = bus_sol.T
        hdata["bus"].y = torch.tensor(
            _align_features(bus_sol, S["H5BusSolution"], S["JSONBusSolution"]),
            dtype=torch.float32,
        )

    if "generator" in sol_nodes:
        gen_sol = sol_nodes["generator"][()]
        if gen_sol.shape[0] == 2:
            gen_sol = gen_sol.T
        hdata["generator"].y = torch.tensor(
            _align_features(
                gen_sol, S["H5GeneratorSolution"], S["JSONGeneratorSolution"]
            ),
            dtype=torch.float32,
        )

    if "load" in sol_nodes:
        load_sol = sol_nodes["load"][()]
        if load_sol.shape[0] == 2:
            load_sol = load_sol.T
        if load_sol.shape[1] == 2:
            hdata["load"].y = torch.tensor(
                _align_features(
                    load_sol,
                    S["ContingencyH5LoadSolution"],
                    S["JSONLoad"],
                ),
                dtype=torch.float32,
            )


def _process_edges_hdf5(hdata: HeteroData, grid, solution):
    """Process edge data from HDF5 format."""
    h5py = _lazy_import_h5py()
    S = _get_schemas()

    if "edges" not in grid:
        return

    edges = grid["edges"]
    sol_edges = solution.get("edges", {})

    # --- ac_line ---
    if "ac_line" in edges:
        ac_edge = edges["ac_line"]
        senders = ac_edge["senders"][()]
        receivers = ac_edge["receivers"][()]
        hdata["bus", "ac_line", "bus"].edge_index = torch.stack(
            [
                torch.tensor(senders, dtype=torch.long),
                torch.tensor(receivers, dtype=torch.long),
            ],
            dim=0,
        )

        ac_features = ac_edge["features"][()]
        if ac_features.shape[0] == 9 or ac_features.shape[0] == 10:
            ac_features = ac_features.T

        if ac_features.shape[0] > 0 and ac_features.shape[1] >= 9:
            ac_x_final = _align_features(
                ac_features, S["H5ACLine"], S["JSONACLine"]
            )
        elif ac_features.shape[0] > 0:
            ac_x_final = ac_features
        else:
            ac_x_final = np.zeros(
                (0, len(S["JSONACLine"].get_feature_names()))
            )

        hdata["bus", "ac_line", "bus"].edge_attr = torch.tensor(
            ac_x_final, dtype=torch.float32
        )

        if "ac_line" in sol_edges:
            ac_sol_obj = sol_edges["ac_line"]
            if isinstance(ac_sol_obj, h5py.Dataset):
                ac_sol = ac_sol_obj[()]
            elif (
                isinstance(ac_sol_obj, h5py.Group)
                and "features" in ac_sol_obj
            ):
                ac_sol = ac_sol_obj["features"][()]
            else:
                ac_sol = None

            if ac_sol is not None:
                if ac_sol.shape[0] == 4:
                    ac_sol = ac_sol.T
                if ac_sol.shape[0] > 0:
                    hdata["bus", "ac_line", "bus"].edge_label = torch.tensor(
                        _align_features(
                            ac_sol,
                            S["H5EdgeSolution"],
                            S["JSONEdgeSolution"],
                        ),
                        dtype=torch.float32,
                    )
                else:
                    hdata["bus", "ac_line", "bus"].edge_label = torch.zeros(
                        (0, len(S["JSONEdgeSolution"].get_feature_names())),
                        dtype=torch.float32,
                    )

    # --- transformer ---
    if "transformer" in edges:
        trans_edge = edges["transformer"]
        senders = trans_edge["senders"][()]
        receivers = trans_edge["receivers"][()]
        hdata["bus", "transformer", "bus"].edge_index = torch.stack(
            [
                torch.tensor(senders, dtype=torch.long),
                torch.tensor(receivers, dtype=torch.long),
            ],
            dim=0,
        )

        trans_features = trans_edge["features"][()]
        if trans_features.shape[0] == 11 or trans_features.shape[0] == 12:
            trans_features = trans_features.T

        if trans_features.shape[0] > 0 and trans_features.shape[1] >= 11:
            trans_x_final = _align_features(
                trans_features, S["H5Transformer"], S["JSONTransformer"]
            )
        elif trans_features.shape[0] > 0:
            trans_x_final = trans_features
        else:
            trans_x_final = np.zeros(
                (0, len(S["JSONTransformer"].get_feature_names()))
            )

        hdata["bus", "transformer", "bus"].edge_attr = torch.tensor(
            trans_x_final, dtype=torch.float32
        )

        if "transformer" in sol_edges:
            tr_sol_obj = sol_edges["transformer"]
            if isinstance(tr_sol_obj, h5py.Dataset):
                trans_sol = tr_sol_obj[()]
            elif (
                isinstance(tr_sol_obj, h5py.Group)
                and "features" in tr_sol_obj
            ):
                trans_sol = tr_sol_obj["features"][()]
            else:
                trans_sol = None

            if trans_sol is not None:
                if trans_sol.shape[0] == 4:
                    trans_sol = trans_sol.T
                if trans_sol.shape[0] > 0:
                    hdata[
                        "bus", "transformer", "bus"
                    ].edge_label = torch.tensor(
                        _align_features(
                            trans_sol,
                            S["H5EdgeSolution"],
                            S["JSONEdgeSolution"],
                        ),
                        dtype=torch.float32,
                    )
                else:
                    hdata[
                        "bus", "transformer", "bus"
                    ].edge_label = torch.zeros(
                        (0, len(S["JSONEdgeSolution"].get_feature_names())),
                        dtype=torch.float32,
                    )

    _process_virtual_links_hdf5(hdata, edges)


def _process_virtual_links_hdf5(hdata: HeteroData, edges):
    """Process virtual links from HDF5 format."""
    if "generator_link" in edges:
        gen_link = edges["generator_link"]
        senders = gen_link["senders"][()]
        receivers = gen_link["receivers"][()]
        hdata["generator", "generator_link", "bus"].edge_index = torch.stack(
            [
                torch.tensor(senders, dtype=torch.long),
                torch.tensor(receivers, dtype=torch.long),
            ],
            dim=0,
        )
        hdata["bus", "generator_link", "generator"].edge_index = torch.stack(
            [
                torch.tensor(receivers, dtype=torch.long),
                torch.tensor(senders, dtype=torch.long),
            ],
            dim=0,
        )

    if "load_link" in edges:
        load_link = edges["load_link"]
        senders = load_link["senders"][()]
        receivers = load_link["receivers"][()]
        hdata["load", "load_link", "bus"].edge_index = torch.stack(
            [
                torch.tensor(senders, dtype=torch.long),
                torch.tensor(receivers, dtype=torch.long),
            ],
            dim=0,
        )
        hdata["bus", "load_link", "load"].edge_index = torch.stack(
            [
                torch.tensor(receivers, dtype=torch.long),
                torch.tensor(senders, dtype=torch.long),
            ],
            dim=0,
        )

    if "shunt_link" in edges:
        shunt_link = edges["shunt_link"]
        senders = shunt_link["senders"][()]
        receivers = shunt_link["receivers"][()]
        if len(senders) > 0:
            hdata["shunt", "shunt_link", "bus"].edge_index = torch.stack(
                [
                    torch.tensor(senders, dtype=torch.long),
                    torch.tensor(receivers, dtype=torch.long),
                ],
                dim=0,
            )
            hdata["bus", "shunt_link", "shunt"].edge_index = torch.stack(
                [
                    torch.tensor(receivers, dtype=torch.long),
                    torch.tensor(senders, dtype=torch.long),
                ],
                dim=0,
            )
        else:
            hdata["shunt", "shunt_link", "bus"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            hdata["bus", "shunt_link", "shunt"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )


def process_hdf5_file(h5_file, n_jobs=1):
    """Process a single HDF5 file into a list of HeteroData objects.

    Each scenario in the HDF5 file is converted to a separate
    ``HeteroData`` object. Contingency scenarios produce multiple
    ``HeteroData`` objects (one per converged post-contingency solution).

    Args:
        h5_file (str): Path to the HDF5 file.
        n_jobs (int): Number of jobs for parallel processing.
            Use ``1`` for sequential (default), ``-1`` for all cores.

    Returns:
        List[HeteroData]: List of processed data objects.
    """
    h5py = _lazy_import_h5py()

    with h5py.File(h5_file, "r") as f:
        scenario_keys = list(f.keys())

    if n_jobs == 1:
        data_list = []
        with h5py.File(h5_file, "r") as f:
            for scenario_key in scenario_keys:
                try:
                    scenario = f[scenario_key]
                    hdata = process_hdf5_scenario(scenario, scenario_key)
                    if hdata is not None:
                        if isinstance(hdata, list):
                            data_list.extend(hdata)
                        else:
                            data_list.append(hdata)
                except Exception as e:
                    print(f"Error processing scenario {scenario_key}: {e}")
                    continue
        return data_list
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_hdf5_scenario_from_path)(h5_file, key)
            for key in tqdm.tqdm(scenario_keys, desc="HDF5 scenarios")
        )

        data_list = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, list):
                data_list.extend(item)
            else:
                data_list.append(item)
        return data_list
