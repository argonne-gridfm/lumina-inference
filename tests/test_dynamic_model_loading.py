"""Tests for dynamic model loading across multiple architectures.

Validates that:
1. resolve_hetero_model_type() correctly resolves all supported model types
2. build_hetero_model_spec() produces correct class + kwargs for each type
3. Modeler.load_model() can dynamically load OPFHeteroGNN, RGAT, HEAT, HGT
4. All architectures produce correct output shapes on synthetic data
5. Backward compatibility: configs without "model" field default to HeteroGNN

All tests run entirely offline with synthetic data.
"""

import copy

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from lumina_inference.model.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina_inference.model.registry import (
    HETERO_MODEL_CLASSES,
    build_hetero_model_spec,
    resolve_hetero_model_type,
)
from lumina_inference.modeler import Modeler


# ---------------------------------------------------------------------------
# Helpers – synthetic test data
# ---------------------------------------------------------------------------

NUM_BUSES = 14
NUM_GENERATORS = 5
NUM_LOADS = 11
NUM_SHUNTS = 1
NUM_AC_LINES = 20

BUS_FEATURE_DIM = 4
GEN_FEATURE_DIM = 7
LOAD_FEATURE_DIM = 2
SHUNT_FEATURE_DIM = 2
AC_LINE_EDGE_ATTR_DIM = 9


def _make_metadata():
    """Return graph metadata dict."""
    return {
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
    }


def _make_input_channels():
    """Return input_channels dict."""
    return {
        "bus": BUS_FEATURE_DIM,
        "generator": GEN_FEATURE_DIM,
        "load": LOAD_FEATURE_DIM,
        "shunt": SHUNT_FEATURE_DIM,
    }


def _make_base_config(model_type=None, model_class=None, models_config=None):
    """Build a config_data dict, optionally specifying model type."""
    config = {
        "case_name": "pglib_opf_case14_ieee",
        "metadata": _make_metadata(),
        "input_channels": _make_input_channels(),
        "config": {
            "models": models_config
            or {
                "HeteroGNN": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "backend": "sage",
                }
            }
        },
    }
    if model_type is not None:
        config["model"] = model_type
    if model_class is not None:
        config["model_class"] = model_class
    return config


def _make_fake_hetero_data():
    """Build a single HeteroData sample with realistic shapes."""
    data = HeteroData()

    bus_x = torch.randn(NUM_BUSES, BUS_FEATURE_DIM)
    bus_x[:, 1] = 0.95  # vmin
    bus_x[:, 2] = 1.05  # vmax
    data["bus"].x = bus_x

    gen_x = torch.randn(NUM_GENERATORS, GEN_FEATURE_DIM)
    gen_x[:, 2] = 0.0  # pmin
    gen_x[:, 3] = 1.0  # pmax
    gen_x[:, 5] = -0.5  # qmin
    gen_x[:, 6] = 0.5  # qmax
    data["generator"].x = gen_x

    data["load"].x = torch.randn(NUM_LOADS, LOAD_FEATURE_DIM)
    data["shunt"].x = torch.randn(NUM_SHUNTS, SHUNT_FEATURE_DIM)

    data["bus"].y = torch.randn(NUM_BUSES, 2)
    data["generator"].y = torch.randn(NUM_GENERATORS, 2)

    src = torch.randint(0, NUM_BUSES, (NUM_AC_LINES,))
    dst = torch.randint(0, NUM_BUSES, (NUM_AC_LINES,))
    data["bus", "ac_line", "bus"].edge_index = torch.stack([src, dst])
    data["bus", "ac_line", "bus"].edge_attr = torch.randn(
        NUM_AC_LINES, AC_LINE_EDGE_ATTR_DIM
    )

    gen_src = torch.arange(NUM_GENERATORS)
    gen_dst = torch.randint(0, NUM_BUSES, (NUM_GENERATORS,))
    data["generator", "generator_link", "bus"].edge_index = torch.stack(
        [gen_src, gen_dst]
    )
    data["bus", "generator_link", "generator"].edge_index = torch.stack(
        [gen_dst, gen_src]
    )

    load_src = torch.arange(NUM_LOADS)
    load_dst = torch.randint(0, NUM_BUSES, (NUM_LOADS,))
    data["load", "load_link", "bus"].edge_index = torch.stack(
        [load_src, load_dst]
    )
    data["bus", "load_link", "load"].edge_index = torch.stack(
        [load_dst, load_src]
    )

    shunt_src = torch.arange(NUM_SHUNTS)
    shunt_dst = torch.randint(0, NUM_BUSES, (NUM_SHUNTS,))
    data["shunt", "shunt_link", "bus"].edge_index = torch.stack(
        [shunt_src, shunt_dst]
    )
    data["bus", "shunt_link", "shunt"].edge_index = torch.stack(
        [shunt_dst, shunt_src]
    )

    return data


# ---------------------------------------------------------------------------
# Tests – resolve_hetero_model_type
# ---------------------------------------------------------------------------


class TestResolveHeteroModelType:
    """Test model type resolution from various input formats."""

    def test_resolve_heterognn_from_model_type(self):
        assert resolve_hetero_model_type(model_type="HeteroGNN") == "HeteroGNN"

    def test_resolve_heterognn_case_insensitive(self):
        assert resolve_hetero_model_type(model_type="heterognn") == "HeteroGNN"

    def test_resolve_opfheterognn_alias(self):
        assert (
            resolve_hetero_model_type(model_type="OPFHeteroGNN") == "HeteroGNN"
        )

    def test_resolve_hgt(self):
        assert resolve_hetero_model_type(model_type="HGT") == "HGT"

    def test_resolve_hgt_case_insensitive(self):
        assert resolve_hetero_model_type(model_type="hgt") == "HGT"

    def test_resolve_rgat(self):
        assert resolve_hetero_model_type(model_type="RGAT") == "RGAT"

    def test_resolve_heat(self):
        assert resolve_hetero_model_type(model_type="HEAT") == "HEAT"

    def test_resolve_from_class_path_hgt(self):
        result = resolve_hetero_model_type(
            model_class_path="lumina.model.opf.hetero_model.HGT"
        )
        assert result == "HGT"

    def test_resolve_from_class_path_opfheterognn(self):
        result = resolve_hetero_model_type(
            model_class_path="lumina.model.opf.hetero_model.OPFHeteroGNN"
        )
        assert result == "HeteroGNN"

    def test_class_path_takes_precedence_over_model_type(self):
        """model_class_path is checked first."""
        result = resolve_hetero_model_type(
            model_type="RGAT",
            model_class_path="lumina.model.opf.hetero_model.HGT",
        )
        assert result == "HGT"

    def test_default_when_both_none(self):
        result = resolve_hetero_model_type(
            model_type=None, model_class_path=None, default="HeteroGNN"
        )
        assert result == "HeteroGNN"

    def test_unsupported_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_hetero_model_type(model_type="UnknownModel")

    def test_unsupported_class_path_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_hetero_model_type(
                model_class_path="some.module.UnknownModel"
            )

    def test_empty_string_model_type_uses_default(self):
        result = resolve_hetero_model_type(
            model_type="", default="HeteroGNN"
        )
        assert result == "HeteroGNN"

    def test_whitespace_model_type_uses_default(self):
        result = resolve_hetero_model_type(
            model_type="  ", default="HeteroGNN"
        )
        assert result == "HeteroGNN"


# ---------------------------------------------------------------------------
# Tests – build_hetero_model_spec
# ---------------------------------------------------------------------------


class TestBuildHeteroModelSpec:
    """Test model spec building for each architecture."""

    def test_build_heterognn_spec(self):
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        models_config = {
            "HeteroGNN": {
                "hidden_channels": 32,
                "num_layers": 3,
                "backend": "sage",
            }
        }
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HeteroGNN", metadata, input_channels, models_config
        )
        assert cls is OPFHeteroGNN
        assert kwargs["hidden_channels"] == 32
        assert kwargs["num_layers"] == 3
        assert kwargs["backend"] == "sage"
        assert not fallback

    def test_build_hgt_spec(self):
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        models_config = {
            "HGT": {
                "hidden_channels": 64,
                "num_layers": 4,
                "num_heads": 4,
            }
        }
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HGT", metadata, input_channels, models_config
        )
        assert cls is HGT
        assert kwargs["hidden_channels"] == 64
        assert kwargs["num_layers"] == 4
        assert kwargs["num_heads"] == 4
        assert not fallback

    def test_build_rgat_spec(self):
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        models_config = {
            "RGAT": {
                "hidden_channels": 32,
                "num_layers": 2,
                "num_heads": 2,
            }
        }
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "RGAT", metadata, input_channels, models_config
        )
        assert cls is RGAT
        assert kwargs["num_heads"] == 2
        assert not fallback

    def test_build_heat_spec(self):
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        models_config = {
            "HEAT": {
                "hidden_channels": 32,
                "num_layers": 2,
                "attention_heads": 2,
            }
        }
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HEAT", metadata, input_channels, models_config
        )
        assert cls is HEAT
        assert kwargs["attention_heads"] == 2
        assert not fallback

    def test_fallback_to_heterognn_config(self):
        """When HGT config is missing, falls back to HeteroGNN config."""
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        models_config = {
            "HeteroGNN": {
                "hidden_channels": 16,
                "num_layers": 2,
                "backend": "sage",
            }
        }
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HGT", metadata, input_channels, models_config
        )
        assert cls is HGT
        assert kwargs["hidden_channels"] == 16
        assert fallback is True

    def test_defaults_applied_when_no_config(self):
        """When models_config is empty, defaults are applied."""
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HeteroGNN", metadata, input_channels, {}
        )
        assert kwargs["hidden_channels"] == 64
        assert kwargs["num_layers"] == 3
        assert kwargs["backend"] == "sage"

    def test_out_channels_passed_through(self):
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HeteroGNN", metadata, input_channels, {}, out_channels=4
        )
        assert kwargs["out_channels"] == 4

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_hetero_model_spec(
                "UnknownModel", _make_metadata(), _make_input_channels(), {}
            )

    def test_hgt_gets_num_heads_default(self):
        """HGT should get num_heads=1 default if not in config."""
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HGT", metadata, input_channels, {}
        )
        assert kwargs["num_heads"] == 1

    def test_heat_gets_attention_heads_default(self):
        """HEAT should get attention_heads=1 default if not in config."""
        metadata = _make_metadata()
        input_channels = _make_input_channels()
        cls, kwargs, config, fallback = build_hetero_model_spec(
            "HEAT", metadata, input_channels, {}
        )
        assert kwargs["attention_heads"] == 1


# ---------------------------------------------------------------------------
# Tests – HETERO_MODEL_CLASSES registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test the model class registry."""

    def test_registry_contains_all_types(self):
        assert "HeteroGNN" in HETERO_MODEL_CLASSES
        assert "RGAT" in HETERO_MODEL_CLASSES
        assert "HEAT" in HETERO_MODEL_CLASSES
        assert "HGT" in HETERO_MODEL_CLASSES

    def test_registry_maps_to_correct_classes(self):
        assert HETERO_MODEL_CLASSES["HeteroGNN"] is OPFHeteroGNN
        assert HETERO_MODEL_CLASSES["RGAT"] is RGAT
        assert HETERO_MODEL_CLASSES["HEAT"] is HEAT
        assert HETERO_MODEL_CLASSES["HGT"] is HGT


# ---------------------------------------------------------------------------
# Tests – Modeler.load_model() with each architecture
# ---------------------------------------------------------------------------


class TestModelerDynamicLoading:
    """Test Modeler.load_model() with different model architectures."""

    @staticmethod
    def _load_and_predict(config_data):
        """Helper: build model from config, get state_dict, load via Modeler,
        and run a single prediction."""
        # Resolve model type and build model to get a valid state_dict
        model_type = resolve_hetero_model_type(
            model_type=config_data.get("model"),
            model_class_path=config_data.get("model_class"),
            default="HeteroGNN",
        )
        model_class, model_kwargs, _, _ = build_hetero_model_spec(
            model_type=model_type,
            metadata=config_data["metadata"],
            input_channels=config_data["input_channels"],
            models_config=config_data.get("config", {}).get("models", {}),
            out_channels=config_data.get("out_channels", 2),
        )
        ref_model = model_class(**model_kwargs)

        # HEAT uses HEATConv with in_channels=-1 (lazy init).
        # A dummy forward pass is needed to materialize parameters
        # before state_dict() can be called.
        if model_type == "HEAT":
            dummy_batch = Batch.from_data_list([_make_fake_hetero_data()])
            for nt in dummy_batch.node_types:
                if hasattr(dummy_batch[nt], "x") and dummy_batch[nt].x is not None:
                    dummy_batch[nt].x = dummy_batch[nt].x.float()
            for et in dummy_batch.edge_types:
                if hasattr(dummy_batch[et], "edge_attr") and dummy_batch[et].edge_attr is not None:
                    dummy_batch[et].edge_attr = dummy_batch[et].edge_attr.float()
            with torch.no_grad():
                ref_model(
                    dummy_batch.x_dict,
                    dummy_batch.edge_index_dict,
                    dummy_batch.edge_attr_dict if hasattr(dummy_batch, "edge_attr_dict") else None,
                    minmax_scaling=False,
                )

        state_dict = ref_model.state_dict()

        # Load via Modeler
        device = torch.device("cpu")
        modeler = Modeler(device, verbose=False)
        modeler.load_model(copy.deepcopy(config_data), state_dict)

        # Predict on synthetic data
        batch = Batch.from_data_list([_make_fake_hetero_data()])
        preds, _ = modeler.predict_batch(batch, minmax_scaling=False)
        return modeler, preds

    def test_load_heterognn_default(self):
        """Config without 'model' field defaults to OPFHeteroGNN."""
        config = _make_base_config()
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, OPFHeteroGNN)
        assert "bus" in preds
        assert "generator" in preds
        assert preds["bus"].shape == (NUM_BUSES, 2)
        assert preds["generator"].shape == (NUM_GENERATORS, 2)

    def test_load_heterognn_explicit(self):
        """Config with model='HeteroGNN' loads OPFHeteroGNN."""
        config = _make_base_config(model_type="HeteroGNN")
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, OPFHeteroGNN)

    def test_load_hgt(self):
        """Config with model='HGT' loads HGT."""
        config = _make_base_config(
            model_type="HGT",
            models_config={
                "HGT": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "num_heads": 2,
                }
            },
        )
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, HGT)
        assert "bus" in preds
        assert "generator" in preds
        assert preds["bus"].shape == (NUM_BUSES, 2)
        assert preds["generator"].shape == (NUM_GENERATORS, 2)

    def test_load_rgat(self):
        """Config with model='RGAT' loads RGAT."""
        config = _make_base_config(
            model_type="RGAT",
            models_config={
                "RGAT": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "num_heads": 2,
                }
            },
        )
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, RGAT)
        assert "bus" in preds
        assert "generator" in preds
        assert preds["bus"].shape == (NUM_BUSES, 2)
        assert preds["generator"].shape == (NUM_GENERATORS, 2)

    def test_load_heat(self):
        """Config with model='HEAT' loads HEAT."""
        config = _make_base_config(
            model_type="HEAT",
            models_config={
                "HEAT": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "attention_heads": 1,
                }
            },
        )
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, HEAT)
        assert "bus" in preds
        assert "generator" in preds
        assert preds["bus"].shape == (NUM_BUSES, 2)
        assert preds["generator"].shape == (NUM_GENERATORS, 2)

    def test_load_hgt_via_model_class_path(self):
        """Config with model_class path loads HGT."""
        config = _make_base_config(
            model_class="lumina.model.opf.hetero_model.HGT",
            models_config={
                "HGT": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "num_heads": 2,
                }
            },
        )
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, HGT)

    def test_load_hgt_with_heterognn_fallback(self):
        """HGT config missing falls back to HeteroGNN config."""
        config = _make_base_config(
            model_type="HGT",
            models_config={
                "HeteroGNN": {
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "backend": "sage",
                }
            },
        )
        modeler, preds = self._load_and_predict(config)
        assert isinstance(modeler.model, HGT)
        assert "bus" in preds

    def test_model_in_eval_mode(self):
        """All loaded models should be in eval mode."""
        for model_type_name in ["HeteroGNN", "HGT", "RGAT", "HEAT"]:
            models_config = {}
            if model_type_name == "HGT":
                models_config = {
                    "HGT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                    }
                }
            elif model_type_name == "RGAT":
                models_config = {
                    "RGAT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                    }
                }
            elif model_type_name == "HEAT":
                models_config = {
                    "HEAT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "attention_heads": 1,
                    }
                }
            else:
                models_config = {
                    "HeteroGNN": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "backend": "sage",
                    }
                }

            config = _make_base_config(
                model_type=model_type_name, models_config=models_config
            )
            modeler, _ = self._load_and_predict(config)
            assert not modeler.model.training, (
                f"{model_type_name} model should be in eval mode"
            )


# ---------------------------------------------------------------------------
# Tests – minmax_scaling with each architecture
# ---------------------------------------------------------------------------


class TestMinmaxScaling:
    """Test that minmax_scaling works with each architecture."""

    @pytest.mark.parametrize(
        "model_type,models_config",
        [
            (
                "HeteroGNN",
                {
                    "HeteroGNN": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "backend": "sage",
                    }
                },
            ),
            (
                "HGT",
                {
                    "HGT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                    }
                },
            ),
            (
                "RGAT",
                {
                    "RGAT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                    }
                },
            ),
            (
                "HEAT",
                {
                    "HEAT": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "attention_heads": 1,
                    }
                },
            ),
        ],
    )
    def test_minmax_scaling_produces_valid_output(
        self, model_type, models_config
    ):
        """minmax_scaling=True should produce bounded outputs."""
        config = _make_base_config(
            model_type=model_type, models_config=models_config
        )

        # Build model and get state_dict
        resolved = resolve_hetero_model_type(model_type=model_type)
        model_class, model_kwargs, _, _ = build_hetero_model_spec(
            model_type=resolved,
            metadata=config["metadata"],
            input_channels=config["input_channels"],
            models_config=models_config,
        )
        ref_model = model_class(**model_kwargs)

        # HEAT uses lazy init — materialize parameters with a dummy pass
        if resolved == "HEAT":
            dummy_batch = Batch.from_data_list([_make_fake_hetero_data()])
            for nt in dummy_batch.node_types:
                if hasattr(dummy_batch[nt], "x") and dummy_batch[nt].x is not None:
                    dummy_batch[nt].x = dummy_batch[nt].x.float()
            for et in dummy_batch.edge_types:
                if hasattr(dummy_batch[et], "edge_attr") and dummy_batch[et].edge_attr is not None:
                    dummy_batch[et].edge_attr = dummy_batch[et].edge_attr.float()
            with torch.no_grad():
                ref_model(
                    dummy_batch.x_dict,
                    dummy_batch.edge_index_dict,
                    dummy_batch.edge_attr_dict if hasattr(dummy_batch, "edge_attr_dict") else None,
                    minmax_scaling=False,
                )

        state_dict = ref_model.state_dict()

        # Load and predict with scaling
        modeler = Modeler(torch.device("cpu"), verbose=False)
        modeler.load_model(copy.deepcopy(config), state_dict)

        batch = Batch.from_data_list([_make_fake_hetero_data()])
        preds, _ = modeler.predict_batch(batch, minmax_scaling=True)

        assert "bus" in preds
        assert "generator" in preds
        # Voltage magnitude (col 1) should be bounded by vmin/vmax
        vm = preds["bus"][:, 1]
        assert torch.all(vm >= 0.94), f"vm min={vm.min().item()}"
        assert torch.all(vm <= 1.06), f"vm max={vm.max().item()}"


# ---------------------------------------------------------------------------
# Tests – backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure existing configs without model type info still work."""

    def test_legacy_config_without_model_field(self):
        """Config from before dynamic loading (no 'model' key) works."""
        config = {
            "case_name": "pglib_opf_case14_ieee",
            "metadata": _make_metadata(),
            "input_channels": _make_input_channels(),
            "config": {
                "models": {
                    "HeteroGNN": {
                        "hidden_channels": 16,
                        "num_layers": 2,
                        "backend": "sage",
                    }
                }
            },
        }
        # No "model" or "model_class" key

        ref_model = OPFHeteroGNN(
            metadata=config["metadata"],
            input_channels=config["input_channels"],
            hidden_channels=16,
            num_layers=2,
            backend="sage",
        )
        state_dict = ref_model.state_dict()

        modeler = Modeler(torch.device("cpu"), verbose=False)
        modeler.load_model(config, state_dict)

        assert isinstance(modeler.model, OPFHeteroGNN)

        batch = Batch.from_data_list([_make_fake_hetero_data()])
        preds, _ = modeler.predict_batch(batch)
        assert "bus" in preds
        assert "generator" in preds

    def test_string_edge_keys_in_metadata(self):
        """Edge keys stored as strings (from JSON) are converted to tuples."""
        config = _make_base_config()
        # Simulate JSON serialization: edge keys become strings
        config["metadata"]["edges"] = {
            str(k): v for k, v in config["metadata"]["edges"].items()
        }

        ref_model = OPFHeteroGNN(
            metadata=_make_metadata(),
            input_channels=_make_input_channels(),
            hidden_channels=16,
            num_layers=2,
            backend="sage",
        )
        state_dict = ref_model.state_dict()

        modeler = Modeler(torch.device("cpu"), verbose=False)
        modeler.load_model(config, state_dict)
        assert isinstance(modeler.model, OPFHeteroGNN)
