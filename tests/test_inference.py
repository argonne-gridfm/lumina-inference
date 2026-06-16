"""Validation tests for lumina-inference.

Tests that:
1. Model loads successfully via Modeler.load_model()
2. Prediction runs on at least one batch from pglib_opf_case14_ieee
3. predictions_cpu is a non-empty dict containing tensors for 'bus' and 'generator'
4. Output shapes are correct (2 columns each for bus and generator)

All Hugging Face Hub downloads, dataset downloads, and other external
network calls are mocked using pytest / unittest.mock so that tests
run entirely offline.
"""

import json
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch
from torch_geometric.data import HeteroData, Batch

from lumina_inference.modeler import Modeler


# ---------------------------------------------------------------------------
# Helpers – fake artifacts
# ---------------------------------------------------------------------------

NUM_BUSES = 14
NUM_GENERATORS = 5
NUM_LOADS = 11
NUM_SHUNTS = 1
NUM_AC_LINES = 20

# Bus features: [pd, vmin, vmax, ...] – need at least 3 cols for OPF schema
BUS_FEATURE_DIM = 4
# Generator features: [pg, qg, pmin, pmax, qg_status, qmin, qmax] – need ≥7
GEN_FEATURE_DIM = 7
LOAD_FEATURE_DIM = 2
SHUNT_FEATURE_DIM = 2
AC_LINE_EDGE_ATTR_DIM = 9


def _make_fake_config():
    """Return a realistic config_data dict matching LUMINA-2M structure."""
    return {
        "case_name": "pglib_opf_case14_ieee",
        "metadata": {
            "nodes": {
                "bus": BUS_FEATURE_DIM,
                "generator": GEN_FEATURE_DIM,
                "load": LOAD_FEATURE_DIM,
                "shunt": SHUNT_FEATURE_DIM,
            },
            "edges": {
                ("bus", "ac_line", "bus"): AC_LINE_EDGE_ATTR_DIM,
                ("generator", "generator_link", "bus"): 0,
                ("bus", "generator_link", "generator"): 0,
                ("load", "load_link", "bus"): 0,
                ("bus", "load_link", "load"): 0,
                ("shunt", "shunt_link", "bus"): 0,
                ("bus", "shunt_link", "shunt"): 0,
            },
        },
        "input_channels": {
            "bus": BUS_FEATURE_DIM,
            "generator": GEN_FEATURE_DIM,
            "load": LOAD_FEATURE_DIM,
            "shunt": SHUNT_FEATURE_DIM,
        },
        "config": {
            "models": {
                "HGT": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "dropout": 0.0,
                }
            }
        },
    }


def _make_fake_hetero_data():
    """Build a single HeteroData sample with realistic shapes."""
    data = HeteroData()

    # Node features --------------------------------------------------------
    # Bus: vmin/vmax in cols 1,2 so OPF scaling can extract bounds
    bus_x = torch.randn(NUM_BUSES, BUS_FEATURE_DIM)
    bus_x[:, 1] = 0.95  # vmin
    bus_x[:, 2] = 1.05  # vmax
    data["bus"].x = bus_x

    gen_x = torch.randn(NUM_GENERATORS, GEN_FEATURE_DIM)
    gen_x[:, 2] = 0.0   # pmin
    gen_x[:, 3] = 1.0   # pmax
    gen_x[:, 5] = -0.5  # qmin
    gen_x[:, 6] = 0.5   # qmax
    data["generator"].x = gen_x

    data["load"].x = torch.randn(NUM_LOADS, LOAD_FEATURE_DIM)
    data["shunt"].x = torch.randn(NUM_SHUNTS, SHUNT_FEATURE_DIM)

    # Targets (y) ----------------------------------------------------------
    data["bus"].y = torch.randn(NUM_BUSES, 2)
    data["generator"].y = torch.randn(NUM_GENERATORS, 2)

    # Edge indices ---------------------------------------------------------
    # ac_line edges (bus -> bus)
    src = torch.randint(0, NUM_BUSES, (NUM_AC_LINES,))
    dst = torch.randint(0, NUM_BUSES, (NUM_AC_LINES,))
    data["bus", "ac_line", "bus"].edge_index = torch.stack([src, dst])
    data["bus", "ac_line", "bus"].edge_attr = torch.randn(
        NUM_AC_LINES, AC_LINE_EDGE_ATTR_DIM
    )

    # generator_link edges (generator -> bus)
    gen_src = torch.arange(NUM_GENERATORS)
    gen_dst = torch.randint(0, NUM_BUSES, (NUM_GENERATORS,))
    data["generator", "generator_link", "bus"].edge_index = torch.stack(
        [gen_src, gen_dst]
    )
    data["bus", "generator_link", "generator"].edge_index = torch.stack(
        [gen_dst, gen_src]
    )

    # load_link edges (load -> bus)
    load_src = torch.arange(NUM_LOADS)
    load_dst = torch.randint(0, NUM_BUSES, (NUM_LOADS,))
    data["load", "load_link", "bus"].edge_index = torch.stack(
        [load_src, load_dst]
    )
    data["bus", "load_link", "load"].edge_index = torch.stack(
        [load_dst, load_src]
    )

    # shunt_link edges (shunt -> bus)
    shunt_src = torch.arange(NUM_SHUNTS)
    shunt_dst = torch.randint(0, NUM_BUSES, (NUM_SHUNTS,))
    data["shunt", "shunt_link", "bus"].edge_index = torch.stack(
        [shunt_src, shunt_dst]
    )
    data["bus", "shunt_link", "shunt"].edge_index = torch.stack(
        [shunt_dst, shunt_src]
    )

    return data


def _make_fake_batch(batch_size=1):
    """Create a batched HeteroData from fake samples."""
    samples = [_make_fake_hetero_data() for _ in range(batch_size)]
    return Batch.from_data_list(samples)


def _build_model_from_config(config_data):
    """Construct a real (small) HGT from the fake config."""
    from lumina_inference.model.hetero_model import HGT

    hgt_cfg = config_data["config"]["models"]["HGT"]
    return HGT(
        metadata=config_data["metadata"],
        input_channels=config_data["input_channels"],
        hidden_channels=hgt_cfg["hidden_channels"],
        num_layers=hgt_cfg["num_layers"],
        num_heads=hgt_cfg.get("num_heads", 1),
        dropout=hgt_cfg.get("dropout", 0.0),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_config():
    """Module-scoped fake config_data."""
    return _make_fake_config()


@pytest.fixture(scope="module")
def fake_state_dict(fake_config):
    """Module-scoped state dict from a freshly-initialised model."""
    model = _build_model_from_config(fake_config)
    return model.state_dict()


@pytest.fixture(scope="module")
def model_artifacts(fake_config, fake_state_dict):
    """Replaces the old fixture that downloaded from Hugging Face."""
    return fake_config, fake_state_dict


@pytest.fixture(scope="module")
def modeler_with_model(model_artifacts):
    """Create a Modeler instance with the model loaded (no network)."""
    config_data, state_dict = model_artifacts
    device = torch.device("cpu")
    modeler = Modeler(device, verbose=False)
    modeler.load_model(config_data, state_dict)
    return modeler, config_data


@pytest.fixture(scope="module")
def dataset_and_loader():
    """Provide a fake dataset and loader without any network access."""
    fake_samples = [_make_fake_hetero_data() for _ in range(5)]

    # Minimal list-based dataset that quacks like a PyG dataset
    class _FakeDataset(list):
        """List wrapper that behaves enough like an InMemoryDataset."""
        pass

    dataset = _FakeDataset(fake_samples)

    # Use PyG's own DataLoader (imported in the test file under test)
    from torch_geometric.loader import DataLoader as PyGDataLoader

    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, loader


# ---------------------------------------------------------------------------
# Tests – Modeler initialisation
# ---------------------------------------------------------------------------


class TestModelerInit:
    """Test Modeler initialization."""

    def test_modeler_creates_without_error(self):
        """Modeler can be instantiated with just a device."""
        modeler = Modeler(torch.device("cpu"))
        assert modeler.model is None
        assert modeler.config_data is None
        assert modeler.device == torch.device("cpu")

    def test_modeler_has_no_evaluate_method(self):
        """Modeler must NOT have evaluate_from_predictions (evaluation removed)."""
        modeler = Modeler(torch.device("cpu"))
        assert not hasattr(modeler, "evaluate_from_predictions")

    def test_modeler_has_no_build_constraint_evaluator(self):
        """Modeler must NOT have build_constraint_evaluator."""
        modeler = Modeler(torch.device("cpu"))
        assert not hasattr(modeler, "build_constraint_evaluator")

    def test_modeler_has_no_base_mva(self):
        """Modeler must NOT accept base_mva parameter."""
        modeler = Modeler(torch.device("cpu"))
        assert not hasattr(modeler, "base_mva")

    def test_modeler_has_no_slack_bus_indices(self):
        """Modeler must NOT accept slack_bus_indices parameter."""
        modeler = Modeler(torch.device("cpu"))
        assert not hasattr(modeler, "slack_bus_indices")


# ---------------------------------------------------------------------------
# Tests – checkpoint key conversion
# ---------------------------------------------------------------------------


class TestCheckpointKeyConversion:
    """Test checkpoint key conversion logic."""

    def test_simple_key_conversion(self):
        key = "convs.0.convs.<bus___ac_line___bus>.lin_l.weight"
        result = Modeler.convert_checkpoint_key_to_model_key(key)
        assert "('bus', 'ac_line', 'bus')" in result

    def test_no_angle_brackets_unchanged(self):
        key = "lin_dict.bus.weight"
        result = Modeler.convert_checkpoint_key_to_model_key(key)
        assert result == key


# ---------------------------------------------------------------------------
# Tests – model loading (mocked HF download)
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Test model loading from (mocked) Hugging Face artifacts."""

    def test_load_model_returns_model_and_config(self, modeler_with_model):
        modeler, config_data = modeler_with_model
        assert modeler.model is not None
        assert modeler.config_data is not None

    def test_model_is_in_eval_mode(self, modeler_with_model):
        modeler, _ = modeler_with_model
        assert not modeler.model.training

    def test_model_is_hgt(self, modeler_with_model):
        from lumina_inference.model.hetero_model import HGT

        modeler, _ = modeler_with_model
        assert isinstance(modeler.model, HGT)

    def test_hf_hub_download_is_not_called(self, fake_config, fake_state_dict):
        """Ensure the real hf_hub_download is never invoked."""
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            device = torch.device("cpu")
            modeler = Modeler(device, verbose=False)
            modeler.load_model(fake_config, fake_state_dict)
            mock_dl.assert_not_called()


class TestShapeMismatchValidation:
    """Issue #10 — config / checkpoint shape mismatch should raise ValueError
    with a user-friendly message, not a PyTorch RuntimeError."""

    def _build_minimal_config(self):
        return {
            "metadata": {
                "nodes": {"bus": {}, "generator": {}},
                "edges": {
                    ("bus", "to", "bus"): {},
                    ("generator", "to", "bus"): {},
                },
            },
            "input_channels": {"bus": 7, "generator": 4},
            "config": {
                "models": {
                    "HGT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                    }
                }
            },
        }

    def _state_dict_for(self, config):
        import copy

        modeler = Modeler(torch.device("cpu"), verbose=False)
        model, _ = modeler.load_model(copy.deepcopy(config), {})
        return {k: v.clone() for k, v in model.state_dict().items()}

    def test_input_channel_mismatch_raises_valueerror(self):
        import copy

        config = self._build_minimal_config()
        state = self._state_dict_for(config)

        bad = copy.deepcopy(config)
        bad["input_channels"]["bus"] = 8  # stale config

        with pytest.raises(ValueError, match="Config / checkpoint shape mismatch"):
            Modeler(torch.device("cpu"), verbose=False).load_model(bad, state)

    def test_hidden_channels_mismatch_raises_valueerror(self):
        import copy

        config = self._build_minimal_config()
        state = self._state_dict_for(config)

        bad = copy.deepcopy(config)
        bad["config"]["models"]["HGT"]["hidden_channels"] = 32

        with pytest.raises(ValueError, match="shape mismatch"):
            Modeler(torch.device("cpu"), verbose=False).load_model(bad, state)

    def test_matching_shapes_load_ok(self):
        """Same config used to build the checkpoint should load cleanly."""
        config = self._build_minimal_config()
        state = self._state_dict_for(config)

        modeler = Modeler(torch.device("cpu"), verbose=False)
        modeler.load_model(config, state)
        assert modeler.model is not None


class TestHGTMetadataValidation:
    """Issue #10 — HGT() should reject metadata that is missing required
    output node types ('bus' and 'generator')."""

    def test_missing_generator_raises(self):
        from lumina_inference.model.hetero_model import HGT

        with pytest.raises(ValueError, match="generator"):
            HGT(
                metadata=(["bus", "load"], [("bus", "x", "load")]),
                input_channels={"bus": 7, "load": 2},
                hidden_channels=8,
                num_layers=1,
                num_heads=1,
            )

    def test_missing_bus_raises(self):
        from lumina_inference.model.hetero_model import HGT

        with pytest.raises(ValueError, match="bus"):
            HGT(
                metadata=(["generator", "load"], [("generator", "x", "load")]),
                input_channels={"generator": 4, "load": 2},
                hidden_channels=8,
                num_layers=1,
                num_heads=1,
            )

    def test_both_required_nodes_present_constructs_ok(self):
        from lumina_inference.model.hetero_model import HGT

        HGT(
            metadata=(
                ["bus", "generator"],
                [("bus", "to", "bus"), ("generator", "to", "bus")],
            ),
            input_channels={"bus": 7, "generator": 4},
            hidden_channels=8,
            num_layers=1,
            num_heads=1,
        )


class TestFromPretrained:
    """Test Modeler.from_pretrained() with mocked HuggingFace + SafeTensors."""

    def test_from_pretrained_downloads_and_loads(
        self, tmp_path, fake_config, fake_state_dict
    ):
        """Verify from_pretrained wires hf_hub_download + load_file + load_model."""
        from safetensors.torch import save_file

        # Stage real artifacts on disk so the mocked download returns them.
        config_path = tmp_path / "config.json"
        weights_path = tmp_path / "model.safetensors"
        with config_path.open("w") as f:
            # config_data has tuple keys in metadata.edges; coerce to str.
            json.dump(
                {
                    **fake_config,
                    "metadata": {
                        **fake_config["metadata"],
                        "edges": {
                            str(k): v
                            for k, v in fake_config["metadata"]["edges"].items()
                        },
                    },
                },
                f,
            )
        # Ensure all tensors are contiguous (safetensors requirement).
        save_file(
            {k: v.contiguous() for k, v in fake_state_dict.items()},
            str(weights_path),
        )

        def fake_download(*, repo_id, filename, **kwargs):
            if filename == "config.json":
                return str(config_path)
            if filename == "model.safetensors":
                return str(weights_path)
            raise FileNotFoundError(filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_download
        ) as mock_dl:
            modeler = Modeler(torch.device("cpu"), verbose=False)
            result = modeler.from_pretrained("argonne/LUMINA-2M")

        # Returns self for chaining
        assert result is modeler
        # Model was constructed and config_data populated
        assert modeler.model is not None
        assert modeler.config_data is not None
        # Both required files were requested
        assert mock_dl.call_count == 2
        called_filenames = {c.kwargs["filename"] for c in mock_dl.call_args_list}
        assert called_filenames == {"config.json", "model.safetensors"}
        # repo_id forwarded correctly
        for call in mock_dl.call_args_list:
            assert call.kwargs["repo_id"] == "argonne/LUMINA-2M"

    def test_from_pretrained_forwards_revision_and_token(
        self, tmp_path, fake_config, fake_state_dict
    ):
        """revision / token / cache_dir args propagate to hf_hub_download."""
        from safetensors.torch import save_file

        config_path = tmp_path / "config.json"
        weights_path = tmp_path / "model.safetensors"
        with config_path.open("w") as f:
            json.dump(
                {
                    **fake_config,
                    "metadata": {
                        **fake_config["metadata"],
                        "edges": {
                            str(k): v
                            for k, v in fake_config["metadata"]["edges"].items()
                        },
                    },
                },
                f,
            )
        save_file(
            {k: v.contiguous() for k, v in fake_state_dict.items()},
            str(weights_path),
        )

        def fake_download(*, repo_id, filename, **kwargs):
            return str(
                config_path if filename == "config.json" else weights_path
            )

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_download
        ) as mock_dl:
            modeler = Modeler(torch.device("cpu"), verbose=False)
            modeler.from_pretrained(
                "argonne/LUMINA-2M",
                revision="v0.1.0",
                token="hf_FAKE",
                cache_dir=str(tmp_path / "cache"),
            )

        for call in mock_dl.call_args_list:
            assert call.kwargs["revision"] == "v0.1.0"
            assert call.kwargs["token"] == "hf_FAKE"
            assert call.kwargs["cache_dir"] == str(tmp_path / "cache")

    def test_from_pretrained_omits_optional_kwargs_when_none(
        self, tmp_path, fake_config, fake_state_dict
    ):
        """When revision/cache_dir are None they should not be forwarded."""
        from safetensors.torch import save_file

        config_path = tmp_path / "config.json"
        weights_path = tmp_path / "model.safetensors"
        with config_path.open("w") as f:
            json.dump(
                {
                    **fake_config,
                    "metadata": {
                        **fake_config["metadata"],
                        "edges": {
                            str(k): v
                            for k, v in fake_config["metadata"]["edges"].items()
                        },
                    },
                },
                f,
            )
        save_file(
            {k: v.contiguous() for k, v in fake_state_dict.items()},
            str(weights_path),
        )

        def fake_download(*, repo_id, filename, **kwargs):
            return str(
                config_path if filename == "config.json" else weights_path
            )

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_download
        ) as mock_dl:
            modeler = Modeler(torch.device("cpu"), verbose=False)
            modeler.from_pretrained("argonne/LUMINA-2M")

        for call in mock_dl.call_args_list:
            # token is always passed (defaults to None for HF auth lookup)
            assert "token" in call.kwargs
            assert call.kwargs["token"] is None
            # revision and cache_dir omitted entirely when None
            assert "revision" not in call.kwargs
            assert "cache_dir" not in call.kwargs


# ---------------------------------------------------------------------------
# Tests – prediction (all data is synthetic)
# ---------------------------------------------------------------------------


class TestPrediction:
    """Test prediction on synthetic data (no network calls)."""

    def test_predict_batch_returns_dict(
        self, modeler_with_model, dataset_and_loader
    ):
        modeler, _ = modeler_with_model
        _, loader = dataset_and_loader

        batch = next(iter(loader))
        predictions_cpu, batch_cpu = modeler.predict_batch(batch)

        assert isinstance(predictions_cpu, dict)
        assert len(predictions_cpu) > 0

    def test_predictions_contain_bus_and_generator(
        self, modeler_with_model, dataset_and_loader
    ):
        modeler, _ = modeler_with_model
        _, loader = dataset_and_loader

        batch = next(iter(loader))
        predictions_cpu, _ = modeler.predict_batch(batch)

        assert "bus" in predictions_cpu, (
            f"Expected 'bus' in predictions, got keys: "
            f"{list(predictions_cpu.keys())}"
        )
        assert "generator" in predictions_cpu, (
            f"Expected 'generator' in predictions, got keys: "
            f"{list(predictions_cpu.keys())}"
        )

    def test_prediction_shapes(
        self, modeler_with_model, dataset_and_loader
    ):
        modeler, _ = modeler_with_model
        _, loader = dataset_and_loader

        batch = next(iter(loader))
        predictions_cpu, _ = modeler.predict_batch(batch)

        bus_pred = predictions_cpu["bus"]
        gen_pred = predictions_cpu["generator"]

        assert isinstance(bus_pred, torch.Tensor)
        assert isinstance(gen_pred, torch.Tensor)
        assert bus_pred.ndim == 2
        assert gen_pred.ndim == 2
        assert bus_pred.shape[1] == 2, (
            f"Expected bus predictions with 2 columns (va, vm), "
            f"got {bus_pred.shape[1]}"
        )
        assert gen_pred.shape[1] == 2, (
            f"Expected generator predictions with 2 columns (pg, qg), "
            f"got {gen_pred.shape[1]}"
        )

    def test_predictions_are_finite(
        self, modeler_with_model, dataset_and_loader
    ):
        modeler, _ = modeler_with_model
        _, loader = dataset_and_loader

        batch = next(iter(loader))
        predictions_cpu, _ = modeler.predict_batch(batch)

        for key, tensor in predictions_cpu.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), (
                    f"Non-finite values found in predictions['{key}']"
                )

    def test_run_predictions_multiple_batches(
        self, modeler_with_model, dataset_and_loader
    ):
        modeler, _ = modeler_with_model
        _, loader = dataset_and_loader

        preds = modeler.run_predictions(loader, max_batches=3)

        assert len(preds) > 0
        assert len(preds) <= 3

        for predictions_cpu, batch_cpu in preds:
            assert "bus" in predictions_cpu
            assert "generator" in predictions_cpu


# ---------------------------------------------------------------------------
# Tests – no training leakage
# ---------------------------------------------------------------------------


class TestNoTrainingLeakage:
    """Verify no training code has leaked into the inference package."""

    def test_no_evaluator_import(self):
        """ACOPFConstraintEvaluator must not be importable."""
        with pytest.raises(ImportError):
            from lumina_inference.evaluator import ACOPFConstraintEvaluator  # noqa: F401

    def test_no_homo_wrapper_import(self):
        """OPFHomoWrapper must not be importable."""
        with pytest.raises(ImportError):
            from lumina_inference.utils import OPFHomoWrapper  # noqa: F401

    def test_modeler_has_required_methods(self):
        """Modeler has only inference methods."""
        modeler = Modeler(torch.device("cpu"))
        required = [
            "load_model",
            "predict_batch",
            "predict_single",
            "run_predictions",
            "to_float32",
            "convert_checkpoint_key_to_model_key",
            "load_checkpoint_into_model",
            "_validate_batch",
        ]
        for method_name in required:
            assert hasattr(modeler, method_name), (
                f"Modeler missing required method: {method_name}"
            )

    def test_modeler_has_no_training_methods(self):
        """Modeler must not have training/evaluation methods."""
        modeler = Modeler(torch.device("cpu"))
        forbidden = [
            "evaluate_from_predictions",
            "build_constraint_evaluator",
            "derive_voltage_limits",
            "derive_generation_limits",
            "derive_line_params",
            "load_model_from_training_checkpoint",
        ]
        for method_name in forbidden:
            assert not hasattr(modeler, method_name), (
                f"Modeler has forbidden method: {method_name}"
            )
