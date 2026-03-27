"""Validation tests for lumina-inference.

Tests that:
1. Model loads successfully via Modeler.load_model()
2. Prediction runs on at least one batch from pglib_opf_case14_ieee
3. predictions_cpu is a non-empty dict containing tensors for 'bus' and 'generator'
4. Output shapes are correct (2 columns each for bus and generator)

For numerical equivalence testing against lumina-core, set the environment
variable LUMINA_CORE_REFERENCE=1 and ensure lumina-core is installed.
"""

import json
import os

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lumina_inference.modeler import Modeler
from lumina_inference.dataset.opf_dataset import OPFDataset
from lumina_inference.loader.opf_loader import DataLoader


@pytest.fixture(scope="module")
def model_artifacts():
    """Download and cache model artifacts for the test session."""
    config_path = hf_hub_download(
        repo_id="argonne/LUMINA-1B", filename="config.json"
    )
    safetensors_path = hf_hub_download(
        repo_id="argonne/LUMINA-1B", filename="model.safetensors"
    )

    with open(config_path, "r") as f:
        config_data = json.load(f)

    state_dict = load_file(safetensors_path)
    return config_data, state_dict


@pytest.fixture(scope="module")
def modeler_with_model(model_artifacts):
    """Create a Modeler instance with the model loaded."""
    config_data, state_dict = model_artifacts
    device = torch.device("cpu")
    modeler = Modeler(device)
    modeler.load_model(config_data, state_dict)
    return modeler, config_data


@pytest.fixture(scope="module")
def dataset_and_loader(model_artifacts):
    """Load the dataset and create a DataLoader."""
    config_data, _ = model_artifacts
    case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
    dataset = OPFDataset(root="./test_opf_data", case_name=case_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, loader


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


class TestModelLoading:
    """Test model loading from Hugging Face artifacts."""

    def test_load_model_returns_model_and_config(self, modeler_with_model):
        modeler, config_data = modeler_with_model
        assert modeler.model is not None
        assert modeler.config_data is not None

    def test_model_is_in_eval_mode(self, modeler_with_model):
        modeler, _ = modeler_with_model
        assert not modeler.model.training

    def test_model_is_opf_hetero_gnn(self, modeler_with_model):
        from lumina_inference.model.hetero_model import OPFHeteroGNN

        modeler, _ = modeler_with_model
        assert isinstance(modeler.model, OPFHeteroGNN)


class TestPrediction:
    """Test prediction on real data."""

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


class TestNumericalEquivalence:
    """Test numerical equivalence against lumina-core reference.

    These tests only run when LUMINA_CORE_REFERENCE=1 is set and
    lumina-core is installed in the environment.
    """

    @pytest.mark.skipif(
        os.environ.get("LUMINA_CORE_REFERENCE") != "1",
        reason="Set LUMINA_CORE_REFERENCE=1 to run equivalence tests",
    )
    def test_predictions_match_lumina_core(
        self, model_artifacts, dataset_and_loader
    ):
        """Compare predictions from lumina-inference against lumina-core."""
        config_data, state_dict = model_artifacts
        _, loader = dataset_and_loader

        # lumina-inference predictions
        device = torch.device("cpu")
        inf_modeler = Modeler(device)
        inf_modeler.load_model(config_data, state_dict)

        batch = next(iter(loader))
        inf_preds, _ = inf_modeler.predict_batch(batch)

        # lumina-core predictions
        from lumina.evaluator.opf.utils import (
            Modeler as CoreModeler,
        )

        core_modeler = CoreModeler(device, slack_bus_indices="0")
        core_modeler.load_model(config_data, state_dict)

        batch2 = next(iter(loader))
        core_preds, _ = core_modeler.predict_batch(batch2)

        # Compare
        for key in ["bus", "generator"]:
            assert torch.allclose(
                inf_preds[key], core_preds[key], atol=1e-5, rtol=1e-4
            ), (
                f"Predictions for '{key}' differ between lumina-inference "
                f"and lumina-core. Max diff: "
                f"{(inf_preds[key] - core_preds[key]).abs().max().item()}"
            )


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
