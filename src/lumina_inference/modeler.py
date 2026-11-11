"""Inference Modeler — load a trained LUMINA model and run predictions.

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

import ast
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from lumina_inference.dataset.utils import hetero_batch_kwargs
from lumina_inference.dataset.validation import (
    MissingDataError,
    SchemaValidationError,
    detect_schema,
    validate_hetero_data,
)
from lumina_inference.dataset.virtual_node import (
    needs_demand_stats,
    select_demand_stats,
    variant_from_metadata,
)
from lumina_inference.model.hetero_model import HGT


def _safe_edge_attr_dict(batch):
    """Return edge_attr_dict if any edges carry attributes, else None.

    HeteroData.edge_attr_dict is a property that raises KeyError when no
    edge type has an edge_attr tensor, so a plain hasattr check is not
    sufficient.
    """
    try:
        return batch.edge_attr_dict
    except KeyError:
        return None


def _is_stats_pair(value) -> bool:
    """Whether a ``demand_stats`` argument is one case's resolved statistics.

    ``Modeler`` accepts ``demand_stats`` in two shapes — one case's resolved
    ``((p_mean, p_std), (q_mean, q_std))``, or a per-case table keyed by case
    name — and treats them very differently, so the two must never be
    confused. A table is a mapping; a resolved pair is two pairs of numbers.

    Args:
        value: The ``demand_stats`` argument as supplied by the caller.

    Returns:
        bool: True for a resolved pair, False for a mapping, ``None``, or
        anything else (which the table path then rejects with its own error).
    """
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence) or len(value) != 2:
        return False
    return all(
        isinstance(half, Sequence)
        and not isinstance(half, (str, bytes))
        and len(half) == 2
        and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in half)
        for half in value
    )


def _resolve_minmax_scaling(config_data, requested: Optional[bool]) -> bool:
    """Decide whether to min-max scale outputs, preferring the checkpoint's setting.

    Training records its own ``minmax_scaling`` setting under that key, so a
    checkpoint can say how its weights were trained.

    Args:
        config_data (dict): Parsed model configuration.
        requested (Optional[bool]): The caller's explicit choice, or None to
            defer to the checkpoint.

    Returns:
        bool: The setting to pass to the model's forward pass.
    """
    recorded = (config_data or {}).get("minmax_scaling")

    if requested is None:
        if recorded is None:
            warnings.warn(
                "Config does not record 'minmax_scaling', so inference assumes "
                "True (the training default). If these weights were trained "
                "without min-max scaling, pass minmax_scaling=False — otherwise "
                "vm, pg and qg are squashed into bounds the weights never saw.",
                stacklevel=3,
            )
            return True
        return bool(recorded)

    if recorded is not None and bool(recorded) != bool(requested):
        warnings.warn(
            f"Checkpoint was trained with minmax_scaling={bool(recorded)} but "
            f"inference is running with {bool(requested)}. Pass "
            f"minmax_scaling={bool(recorded)} to match the training run.",
            stacklevel=3,
        )
    return bool(requested)


class Modeler:
    """Prediction-only wrapper for OPF model loading and inference.

    This class encapsulates model construction from a Hugging Face config,
    weight loading from SafeTensors state dicts, and batch prediction.

    Args:
        device (torch.device): Device to run model inference on.
        fail_on_missing (bool, optional): If True, raise when expected model
            keys are missing during checkpoint load. Defaults to False.
        verbose (bool, optional): If True, print diagnostic messages during
            checkpoint loading. Defaults to True.
        demand_stats (optional): Overrides the statistics the checkpoint
            carries, in either of two shapes — one case's resolved
            ``((p_mean, p_std), (q_mean, q_std))``, which is used as-is, or a
            ``{case_name: [[p_mean, p_std], [q_mean, q_std]]}`` table, whose
            rows are merged over the checkpoint's own. Needed only to supply
            statistics for a case the config does not record, or to substitute
            your own. See
            :func:`~lumina_inference.dataset.virtual_node.compute_demand_stats`.
        case_name (optional): Which case's row to take from a multi-case
            table. Defaults to the config's own ``case_name``, and is
            unnecessary for a table holding a single case.
        virtual_node (bool, optional): If True, reconstruct the virtual-node
            augmentation from the loaded metadata and apply it to incoming
            graphs. Defaults to True; it is a no-op for checkpoints trained
            without a virtual node.

    Attributes:
        device (torch.device): See Args.
        fail_on_missing (bool): See Args.
        verbose (bool): See Args.
        model (Optional[torch.nn.Module]): Loaded model (None until
            ``load_model`` is called).
        config_data (Optional[dict]): Parsed model configuration loaded
            during ``load_model``.
        minmax_scaling (Optional[bool]): The setting resolved from the config
            during ``load_model``, used whenever a caller does not pass one.
        demand_stats_in_use (Optional[tuple]): The resolved
            ``((p_mean, p_std), (q_mean, q_std))`` actually applied to the
            virtual node, or None when the checkpoint needs none.

    Example:
        >>> modeler = Modeler(torch.device("cpu"))
        >>> config = json.load(open("config.json"))
        >>> state_dict = load_file("model.safetensors")
        >>> modeler.load_model(config, state_dict)
        >>> loader = DataLoader(dataset, batch_size=1)
        >>> preds = modeler.run_predictions(loader, max_batches=10)
    """

    def __init__(
        self,
        device: torch.device,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
        demand_stats=None,
        case_name: Optional[str] = None,
        virtual_node: bool = True,
    ):
        self.device = device
        self.fail_on_missing = fail_on_missing
        self.verbose = verbose
        self.demand_stats = demand_stats
        self.case_name = case_name
        self.virtual_node = virtual_node
        self.model: Optional[torch.nn.Module] = None
        self.config_data = None
        self.minmax_scaling: Optional[bool] = None
        self.demand_stats_in_use = None
        self._vn_variant = None

    # -- checkpoint key conversion and loading --------------------------------

    @staticmethod
    def convert_checkpoint_key_to_model_key(key: str) -> str:
        """Convert checkpoint keys to model keys.

        Transforms underscore-delimited items inside angle brackets to
        tuple string representation.

        Args:
            key: Current key with triple underscore delimiters inside
                angle brackets.

        Returns:
            String with angle bracket contents converted to tuple
            representation.

        Example:
            >>> Modeler.convert_checkpoint_key_to_model_key(
            ...     "<bus___ac_line___weight>"
            ... )
            "('bus', 'ac_line', 'weight')"
        """
        pattern = r"<([^>]+)>"

        def replacer(match):
            parts = match.group(1).split("___")
            return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"

        return re.sub(pattern, replacer, key)

    def load_checkpoint_into_model(
        self,
        model: torch.nn.Module,
        checkpoint_dict,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
    ):
        """Load a checkpoint dictionary into a model.

        Remaps checkpoint keys to model keys using
        ``convert_checkpoint_key_to_model_key`` and then calls
        ``load_state_dict`` with ``strict=False``.

        Args:
            model (torch.nn.Module): The model to populate.
            checkpoint_dict (dict): Mapping of checkpoint keys to tensors.
            fail_on_missing (bool, optional): If True, raise ValueError
                when missing keys remain. Defaults to False.
            verbose (bool, optional): If True, print missing/unexpected keys.

        Returns:
            dict: A dictionary with keys ``"missing_keys"`` and
            ``"unexpected_keys"``.

        Raises:
            ValueError: If ``fail_on_missing`` is True and missing keys
                are found.
        """
        model_state = model.state_dict()
        used_keys = set()

        remapped_state = {}
        for model_key in model_state.keys():
            ck = self.convert_checkpoint_key_to_model_key(model_key)
            if ck in checkpoint_dict:
                remapped_state[model_key] = checkpoint_dict[ck]
                used_keys.add(ck)

        unexpected_keys = [
            k for k in checkpoint_dict.keys() if k not in used_keys
        ]

        load_result = model.load_state_dict(remapped_state, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys.extend(list(load_result.unexpected_keys))

        if verbose and (missing_keys or unexpected_keys):
            print(
                f"[CHECKPOINT LOAD] Missing keys: {missing_keys}, "
                f"Unexpected keys: {unexpected_keys}"
            )
        if fail_on_missing and missing_keys:
            raise ValueError(f"Missing keys during load: {missing_keys}")

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

    # -- model loading --------------------------------------------------------

    def load_model(self, config_data: dict, state_dict: dict):
        """Construct the HGT model from config and state dict.

        Args:
            config_data (dict): Parsed JSON configuration describing model
                metadata and architecture.
            state_dict (dict): Raw state dictionary as returned by
                ``safetensors.torch.load_file``.

        Returns:
            Tuple[torch.nn.Module, dict]: The constructed model (in eval
            mode) and the config_data used to build it.

        Raises:
            ValueError: If ``fail_on_missing`` is True and required keys
                are missing from the checkpoint.
        """
        # Convert metadata edge keys from strings to tuples if needed
        if "edges" in config_data.get("metadata", {}):
            edges_dict = {}
            for key, value in config_data["metadata"]["edges"].items():
                if isinstance(key, str) and key.startswith("("):
                    key = ast.literal_eval(key)
                edges_dict[key] = value
            config_data["metadata"]["edges"] = edges_dict

        models_cfg = config_data["config"]["models"]
        hgt_cfg = models_cfg.get("HGT") or models_cfg.get("HeteroGNN")
        if hgt_cfg is None:
            raise KeyError(
                "config['models'] must contain an 'HGT' or 'HeteroGNN' key"
            )
        # Splat the whole architecture config, as training does, so that a
        # checkpoint's hyperparameters reach the constructor whether or not
        # this file knows their names. HGT absorbs unknown keys via **kwargs.
        model_kwargs = dict(hgt_cfg)
        model_kwargs.update(
            metadata=config_data["metadata"],
            input_channels=config_data["input_channels"],
            out_channels=config_data.get("out_channels", 2),
        )
        model_kwargs.setdefault("hidden_channels", 64)
        model_kwargs.setdefault("num_layers", 3)
        model_kwargs.setdefault("num_heads", 1)
        model = HGT(**model_kwargs).to(self.device)

        # state_dict is the raw output of safetensors.load_file; remap keys
        checkpoint_dict = {
            self.convert_checkpoint_key_to_model_key(k): v
            for k, v in state_dict.items()
        }

        self.load_checkpoint_into_model(
            model,
            checkpoint_dict,
            fail_on_missing=self.fail_on_missing,
            verbose=self.verbose,
        )

        model.eval()
        self.model = model
        self.config_data = config_data

        # Resolve once, so the "assuming True" warning fires at load time
        # rather than on every batch.
        self.minmax_scaling = _resolve_minmax_scaling(config_data, None)

        # Resolve only when the virtual node actually standardises demand. A
        # run's config keeps its demand_stats table whatever variant it used,
        # so a width-1 'grid' node must not be held to a lookup -- let alone an
        # ambiguous one -- for statistics it never consumes.
        stats = None
        if self.virtual_node and needs_demand_stats(config_data["metadata"]):
            stats = self._resolve_demand_stats(config_data)
            if stats is None:
                raise ValueError(
                    "this checkpoint expects a virtual node carrying "
                    "standardised demand, but no demand statistics are "
                    "available for it. A checkpoint normally records them per "
                    "case under config['virtual_node']['demand_stats']; this "
                    "config carries none, so pass "
                    "demand_stats=((p_mean, p_std), (q_mean, q_std)) for the "
                    "case being served, e.g. from "
                    "compute_demand_stats(samples). Without them the "
                    "standardised channels would be constant zero, so the "
                    "virtual node would not carry the demand signal the "
                    "weights expect."
                )
        self.demand_stats_in_use = stats

        self._vn_variant = (
            variant_from_metadata(config_data["metadata"], stats)
            if self.virtual_node
            else None
        )
        if self.verbose and self._vn_variant is not None:
            print(
                f"[MODEL LOAD] Checkpoint expects a virtual node; input graphs "
                f"will be augmented with {self._vn_variant!r}"
            )
            if stats is not None:
                (p_mean, p_std), (q_mean, q_std) = stats
                source = (
                    "caller-supplied statistics"
                    if self.demand_stats is not None
                    else "statistics from the checkpoint config"
                )
                case = self.case_name or config_data.get("case_name")
                print(
                    f"[MODEL LOAD] Standardising total demand against "
                    f"{source}"
                    + (f" for {case}" if case else "")
                    + f": Pd ~ ({p_mean:.6g}, {p_std:.6g}), "
                    f"Qd ~ ({q_mean:.6g}, {q_std:.6g})"
                )

        return model, config_data

    def _resolve_demand_stats(self, config_data: dict):
        """Resolve the demand statistics to standardise the virtual node with.

        A checkpoint records them per case under
        ``config['virtual_node']['demand_stats']``, because total demand scales
        with the network. The caller's ``demand_stats`` overrides that —
        outright when it is one case's resolved pair, or row by row when it is
        a table of its own.

        Args:
            config_data (dict): Parsed model configuration.

        Returns:
            Optional[tuple]: ``((p_mean, p_std), (q_mean, q_std))``, or None
            when neither the caller nor the config supplies any.

        Raises:
            ValueError: If a case has to be chosen and cannot be — see
                :func:`~lumina_inference.dataset.virtual_node.select_demand_stats`.
        """
        override = self.demand_stats
        if _is_stats_pair(override):
            (p_mean, p_std), (q_mean, q_std) = override
            return ((float(p_mean), float(p_std)), (float(q_mean), float(q_std)))

        virtual_node_cfg = config_data.get("virtual_node") or {}
        table = dict(virtual_node_cfg.get("demand_stats") or {})
        if isinstance(override, Mapping):
            table.update(override)
        elif override is not None:
            raise ValueError(
                f"demand_stats must be either one case's ((p_mean, p_std), "
                f"(q_mean, q_std)) or a {{case_name: [[p_mean, p_std], "
                f"[q_mean, q_std]]}} table, got {type(override).__name__}"
            )

        return select_demand_stats(
            table, self.case_name or config_data.get("case_name")
        )

    @property
    def virtual_node_transform(self):
        """The augmentation this checkpoint's graphs need, if any.

        ``predict_single`` applies this automatically. The batched pipeline
        collates before prediction, while the star edges have to be built
        per-sample, so pass this to the dataset instead::

            dataset = OPFDataset(root, case_name,
                                 transform=modeler.virtual_node_transform)

        Returns:
            Optional[Variant]: A callable taking and returning a
            ``HeteroData``, or ``None`` for a checkpoint trained without a
            virtual node.
        """
        return self._vn_variant

    # -- helpers for tensors --------------------------------------------------

    @staticmethod
    def to_float32(batch):
        """Convert node features/targets and edge attributes to float32.

        Iterates through all node types and edge types in the batch,
        converting ``x`` (features), ``y`` (targets), and ``edge_attr``
        tensors to float32 precision.

        Args:
            batch: A batch data object from DataLoader.

        Returns:
            The same batch object with tensors converted to float32.
        """
        for node_type in batch.node_types:
            if getattr(batch[node_type], "x", None) is not None:
                batch[node_type].x = batch[node_type].x.float()
            if getattr(batch[node_type], "y", None) is not None:
                batch[node_type].y = batch[node_type].y.float()

        for edge_type in batch.edge_types:
            if getattr(batch[edge_type], "edge_attr", None) is not None:
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()

        return batch

    # -- prediction -----------------------------------------------------------

    def predict_batch(self, batch, minmax_scaling: Optional[bool] = None):
        """Run a forward pass on a single batch.

        Args:
            batch: Batched data object containing inputs for the model.
            minmax_scaling (Optional[bool], optional): Whether to apply
                min-max scaling in the model's forward pass. Defaults to
                None, meaning use the setting recorded in the config, which
                is True when the config does not record one.

        Returns:
            Tuple[dict, object]: A tuple ``(predictions_cpu, batch_cpu)``
            where ``predictions_cpu`` maps output names to CPU tensors
            and ``batch_cpu`` is the input batch moved to CPU.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
            SchemaValidationError: If the model expects a virtual node that
                the batch does not carry.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._require_virtual_node(batch)
        batch = self.to_float32(batch).to(self.device)

        predictions = self.model(
            batch.x_dict,
            batch.edge_index_dict,
            _safe_edge_attr_dict(batch),
            minmax_scaling=self._minmax(minmax_scaling),
            **hetero_batch_kwargs(batch),
        )

        # Move predictions to CPU and detach
        predictions_cpu = {}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                predictions_cpu[k] = v.detach().cpu()
            else:
                predictions_cpu[k] = v

        batch_cpu = batch.to(torch.device("cpu"))
        return predictions_cpu, batch_cpu

    def run_predictions(
        self,
        loader: Iterable,
        max_batches: Optional[int] = None,
        minmax_scaling: Optional[bool] = None,
    ):
        """Run predictions over a data loader.

        This method separates the forward pass from evaluation so
        predictions can be stored or evaluated later.

        Args:
            loader (Iterable): Iterable data loader yielding batches.
            max_batches (Optional[int], optional): Limit on number of
                batches to process. Defaults to None (process all).
            minmax_scaling (Optional[bool], optional): Passed to
                ``predict_batch``. Defaults to None (use the config's
                setting).

        Returns:
            List[Tuple[dict, object]]: List of
            ``(predictions_cpu, batch_cpu)`` tuples.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_batch_pairs: List[Tuple[dict, object]] = []
        total_batches = None
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and max_batches is not None:
            total_batches = min(total_batches, max_batches)

        progress_iter = tqdm(
            loader, total=total_batches, desc="Predicting samples"
        )
        for batch_idx, batch in enumerate(progress_iter):
            preds, batch_cpu = self.predict_batch(
                batch, minmax_scaling=minmax_scaling
            )
            pred_batch_pairs.append((preds, batch_cpu))

            progress_iter.set_postfix(
                predictions=len(pred_batch_pairs), refresh=False
            )
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                progress_iter.write(f"Reached max_batches={max_batches}.")
                break
        progress_iter.close()
        return pred_batch_pairs

    # -- single-sample prediction ---------------------------------------------

    def predict_single(
        self,
        data: HeteroData,
        minmax_scaling: Optional[bool] = None,
        validate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a single HeteroData object.

        Wraps the data in a batch of size 1 and delegates to
        :meth:`predict_batch`. This is the primary entry point for
        flexible ingestion — users can pass HeteroData from any of the
        ``load_from_*`` functions or :func:`build_hetero_data`.

        Args:
            data (HeteroData): A single heterogeneous graph data object.
            minmax_scaling (Optional[bool], optional): Whether to apply
                min-max scaling in the model's forward pass. Defaults to
                None, meaning use the setting recorded in the config, which
                is True when the config does not record one.
            validate (bool, optional): Whether to validate the data
                against the loaded model's schema before prediction.
                Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: Predictions keyed by output node
            type (e.g., ``'bus'``, ``'generator'``), with tensors on CPU.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
            SchemaValidationError: If validation is enabled and the data
                doesn't match the model's expected schema.
            FeatureDimensionError: If validation is enabled and feature
                dimensions don't match.
            MissingDataError: If validation is enabled and required
                tensors are missing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Augment before validating: the model's metadata requires the virtual
        # node, so validation would otherwise reject un-augmented input. The
        # transform shallow-copies, so the caller's graph is left untouched.
        if self._vn_variant is not None:
            data = self._vn_variant(data)

        if validate:
            self._validate_batch(data)

        from torch_geometric.data import Batch

        batch = Batch.from_data_list([data])
        predictions_cpu, _ = self.predict_batch(
            batch, minmax_scaling=minmax_scaling
        )
        return predictions_cpu

    # -- validation -----------------------------------------------------------

    def _minmax(self, requested: Optional[bool]) -> bool:
        """Resolve the min-max setting for one prediction call.

        Args:
            requested (Optional[bool]): The caller's explicit choice, or None
                to use the setting resolved at load time.

        Returns:
            bool: The setting to pass to the model's forward pass.
        """
        if requested is None:
            # Already resolved (and warned about) in load_model.
            return bool(self.minmax_scaling)
        return _resolve_minmax_scaling(self.config_data, requested)

    def _require_virtual_node(self, batch) -> None:
        """Fail clearly when a virtual-node model is fed un-augmented data.

        Args:
            batch: The batch about to be predicted on.

        Raises:
            SchemaValidationError: If the model expects a virtual node that
                the batch does not carry.
        """
        if self._vn_variant is None:
            return
        node_type = self._vn_variant.virtual_node_type
        if node_type in batch.node_types:
            return
        raise SchemaValidationError(
            f"This checkpoint was trained with a {node_type!r} virtual node, "
            f"but the batch has none. Batches are collated before prediction "
            f"while the virtual node must be added per-sample, so apply the "
            f"augmentation in the dataset: "
            f"OPFDataset(..., transform=modeler.virtual_node_transform).",
            expected=f"node type {node_type!r}",
            actual=str(sorted(batch.node_types)),
        )

    def _validate_batch(self, data: HeteroData) -> None:
        """Validate a HeteroData object against the loaded model's schema.

        Checks that the data has all node types and feature dimensions
        expected by the model. Uses the model's ``config_data`` to
        determine expected types and dimensions.

        Args:
            data (HeteroData): Data to validate.

        Raises:
            RuntimeError: If the model has not been loaded.
            SchemaValidationError: If structural requirements are not met.
            FeatureDimensionError: If feature dimensions don't match.
            MissingDataError: If required tensors are missing.
        """
        if self.model is None or self.config_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        model_metadata = self.config_data.get("metadata")
        model_input_channels = self.config_data.get("input_channels")

        validate_hetero_data(
            data,
            model_metadata=model_metadata,
            model_input_channels=model_input_channels,
        )
