"""Tests for the release tooling (model card + uploader staging).

Network / Hugging Face Hub interactions are not exercised; we only
verify model-card rendering, release-history construction, and the
staging-only path of the uploader. Tag creation, Collection joins, and
the actual upload call are tested with mocks (see ``test_upload_with_mocked_hub``).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from lumina_inference.modeler import Modeler
from lumina_inference.release import (
    ReleaseHistoryEntry,
    build_history,
    format_parameter_count,
    generate_model_card,
    parse_history,
    render_history,
)
from lumina_inference.release.model_card import (
    _format_input_channels,
    count_parameters,
)
from lumina_inference.release.uploader import HFUploader


# ---------------------------------------------------------------------------
# Fixtures – minimal fake config / state_dict
# ---------------------------------------------------------------------------

BUS_FEATURE_DIM = 4
GEN_FEATURE_DIM = 7
LOAD_FEATURE_DIM = 2
SHUNT_FEATURE_DIM = 2
AC_LINE_EDGE_ATTR_DIM = 9


def _fake_config():
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
                    "hidden_channels": 8,
                    "num_layers": 1,
                    "num_heads": 1,
                    "dropout": 0.0,
                }
            }
        },
    }


def _build_loaded_modeler():
    """Construct a real (tiny) HGT and load it into a Modeler."""
    from lumina_inference.model.hetero_model import HGT

    config_data = _fake_config()
    hgt_cfg = config_data["config"]["models"]["HGT"]
    model = HGT(
        metadata=config_data["metadata"],
        input_channels=config_data["input_channels"],
        hidden_channels=hgt_cfg["hidden_channels"],
        num_layers=hgt_cfg["num_layers"],
        num_heads=hgt_cfg["num_heads"],
        dropout=hgt_cfg["dropout"],
    )
    state_dict = model.state_dict()

    modeler = Modeler(torch.device("cpu"), verbose=False)
    modeler.load_model(config_data, state_dict)
    return modeler, config_data, state_dict


def _fake_checkpoint(config_data, state_dict):
    """Pre-normalized checkpoint schema (legacy / test-friendly).

    This is what an external maintainer might hand-craft before passing
    to the uploader. The uploader's ``_normalize_checkpoint`` accepts
    it as a fast-path (``config_data`` already present).
    """
    return {
        "config_data": config_data,
        "model_state_dict": state_dict,
        "case_name": "pglib_opf_case14_ieee",
        "epoch": 20,
        "val_loss": 0.082726,
        "timestamp": "20251112_193637",
        "model_name": "models.HGT",
        "input_channels": config_data["input_channels"],
    }


def _fake_lumina_checkpoint(config_data, state_dict):
    """Real lumina-core training-checkpoint schema.

    Mirrors the structure observed in actual checkpoints captured
    under ``data/eagle/``: ``model_kwargs`` holds the *real* model
    constructor arguments, ``config[models][HGT]`` may carry stale or
    different hyperparameters, and provenance uses plural
    ``case_names`` + ``best_val_loss``.
    """
    hgt_cfg = config_data["config"]["models"]["HGT"]
    return {
        "model_state_dict": state_dict,
        "model_class": "lumina.model.opf.hetero_model.HGT",
        "model_kwargs": {
            "metadata": config_data["metadata"],
            "input_channels": config_data["input_channels"],
            "hidden_channels": hgt_cfg["hidden_channels"],
            "num_layers": hgt_cfg["num_layers"],
            "num_heads": hgt_cfg["num_heads"],
            "backend": "sage",
        },
        "config": {
            "models": {
                "HGT": hgt_cfg,
                "HeteroGNN": {"hidden_channels": 256, "num_layers": 6},
            },
            # Other sections present in real checkpoints, ignored by uploader
            "training": {"batch_size": 32},
        },
        "case_names": [
            "pglib_opf_case30_ieee",
            "pglib_opf_case57_ieee",
            "pglib_opf_case118_ieee",
        ],
        "best_val_loss": 0.012242,
        "epoch": 27,
        "loss_type": "augmented_lagrangian",
        "run_metadata": {"world_size": 4},
    }


# ---------------------------------------------------------------------------
# format_parameter_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (500, "500"),
        (2_345, "2.3K"),
        (2_345_678, "2.3M"),
        (1_006_940_164, "1.0B"),
    ],
)
def test_format_parameter_count(n, expected):
    assert format_parameter_count(n) == expected


# ---------------------------------------------------------------------------
# _format_input_channels
# ---------------------------------------------------------------------------


def test_format_input_channels_int_mapping():
    out = _format_input_channels({"bus": 7, "generator": 11})
    assert "- **Bus**: 7 features" in out
    assert "- **Generator**: 11 features" in out


def test_format_input_channels_empty():
    assert "No input channel metadata" in _format_input_channels({})


def test_format_input_channels_list_uses_length():
    out = _format_input_channels({"bus": ["a", "b", "c"]})
    assert "- **Bus**: 3 features" in out


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


def test_count_parameters_matches_model():
    modeler, _, _ = _build_loaded_modeler()
    total, trainable = count_parameters(modeler.model)
    assert total > 0
    assert total == trainable  # nothing frozen in our fake model


# ---------------------------------------------------------------------------
# generate_model_card
# ---------------------------------------------------------------------------


def test_generate_model_card_renders_all_placeholders():
    modeler, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)

    card = generate_model_card(
        modeler,
        checkpoint,
        model_name="LUMINA-TEST",
        hf_repo_id="argonne/LUMINA-TEST",
        model_version="v0.1.0",
        commit_sha="abcdef1",
        release_history="_No prior releases recorded._",
    )

    assert "LUMINA-TEST" in card
    assert "argonne/LUMINA-TEST" in card
    assert "HGT" in card
    assert "0.082726" in card
    assert "20251112_193637" in card
    assert "pglib_opf_case14_ieee" in card
    assert "v0.1.0" in card
    assert "abcdef1" in card
    assert "license: other" in card


def test_generate_model_card_unloaded_modeler_raises():
    modeler = Modeler(torch.device("cpu"), verbose=False)
    with pytest.raises(RuntimeError, match="Modeler.model is None"):
        generate_model_card(
            modeler,
            {},
            model_name="X",
            hf_repo_id="org/X",
        )


def test_generate_model_card_handles_missing_val_loss():
    modeler, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint.pop("val_loss")
    card = generate_model_card(
        modeler,
        checkpoint,
        model_name="LUMINA-TEST",
        hf_repo_id="org/LUMINA-TEST",
    )
    assert "Final Validation Loss**: Unknown" in card


def test_generate_model_card_uses_default_release_history_placeholder():
    modeler, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    card = generate_model_card(
        modeler,
        checkpoint,
        model_name="LUMINA-TEST",
        hf_repo_id="org/LUMINA-TEST",
    )
    assert "No prior releases recorded" in card


# ---------------------------------------------------------------------------
# release_history
# ---------------------------------------------------------------------------


def test_render_history_empty():
    assert render_history([]) == "_No prior releases recorded._"


def test_render_history_single_entry():
    entry = ReleaseHistoryEntry(
        version="v0.1.0",
        date="20251112",
        epochs="20",
        val_loss="0.083",
        training_case="OPFData-All",
        commit="a1b2c3d",
    )
    out = render_history([entry])
    assert "| Version | Date" in out
    assert "| v0.1.0 | 20251112 | 20 | 0.083 | OPFData-All | a1b2c3d |" in out


def test_parse_history_roundtrip():
    entries = [
        ReleaseHistoryEntry("v0.2.0", "20260103", "30", "0.071", "All", "7f3a2b1"),
        ReleaseHistoryEntry("v0.1.0", "20251112", "20", "0.083", "All", "a1b2c3d"),
    ]
    markdown = (
        "# Some Card\n\n## Release History\n\n"
        + render_history(entries)
        + "\n\n## Acknowledgements\n"
    )
    parsed = parse_history(markdown)
    assert [e.version for e in parsed] == ["v0.2.0", "v0.1.0"]
    assert parsed[0].commit == "7f3a2b1"


def test_parse_history_no_section_returns_empty():
    assert parse_history("# Card with no history\n") == []


def test_build_history_prepends_new_release():
    prior = [
        ReleaseHistoryEntry("v0.1.0", "20251112", "20", "0.083", "All", "a1b2c3d"),
    ]
    existing = (
        "# Card\n\n## Release History\n\n"
        + render_history(prior)
        + "\n\n## End\n"
    )
    new_entry = ReleaseHistoryEntry(
        "v0.2.0", "20260103", "30", "0.071", "All", "pending"
    )
    out = build_history(new_entry, existing_readme=existing)
    # New entry must come first
    new_idx = out.index("v0.2.0")
    old_idx = out.index("v0.1.0")
    assert new_idx < old_idx


def test_build_history_idempotent_on_same_version():
    """Re-uploading the same version replaces the entry, doesn't duplicate."""
    prior = [
        ReleaseHistoryEntry("v0.1.0", "20251112", "20", "0.083", "All", "old"),
    ]
    existing = (
        "# Card\n\n## Release History\n\n"
        + render_history(prior)
        + "\n\n## End\n"
    )
    new_entry = ReleaseHistoryEntry(
        "v0.1.0", "20251115", "21", "0.080", "All", "new"
    )
    out = build_history(new_entry, existing_readme=existing)
    assert out.count("v0.1.0") == 1
    assert "0.080" in out
    assert "0.083" not in out


def test_build_history_no_existing_readme():
    new_entry = ReleaseHistoryEntry(
        "v0.1.0", "20251112", "20", "0.083", "All", "pending"
    )
    out = build_history(new_entry, existing_readme=None)
    assert "v0.1.0" in out
    assert "| Version | Date" in out


# ---------------------------------------------------------------------------
# Version validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version", ["v0.1.0", "v1.0.0", "v0.2.0-rc1", "v10.20.30"]
)
def test_version_validation_accepts_valid(version):
    HFUploader._validate_version(version)  # no raise


@pytest.mark.parametrize(
    "version", ["0.1.0", "v0.1", "v1", "release-1", "v0.1.0.0"]
)
def test_version_validation_rejects_invalid(version):
    with pytest.raises(ValueError, match="Invalid --version"):
        HFUploader._validate_version(version)


# ---------------------------------------------------------------------------
# HFUploader.stage_artifacts (no network)
# ---------------------------------------------------------------------------


def test_stage_artifacts_writes_expected_files(tmp_path: Path):
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    staging = tmp_path / "release"
    uploader = HFUploader(repo_id="argonne/LUMINA-TEST")
    result = uploader.stage_artifacts(
        checkpoint_path, staging, version="v0.1.0"
    )

    assert (staging / "model.pt").is_file()
    assert (staging / "config.json").is_file()
    assert (staging / "model.safetensors").is_file()
    assert (staging / "requirements.txt").is_file()
    assert (staging / "README.md").is_file()
    assert (staging / "LICENSE").is_file()

    # LICENSE content sanity check (placeholder; final license is TBD)
    license_text = (staging / "LICENSE").read_text()
    assert "LICENSE — TBD" in license_text

    # config.json is the *normalized* config_data: contains 'metadata',
    # 'input_channels', and the 'config.models' arch slice. It does
    # NOT contain training provenance (epoch / case_name / etc).
    cfg = json.loads((staging / "config.json").read_text())
    assert "input_channels" in cfg
    assert "metadata" in cfg
    assert "HGT" in cfg["config"]["models"]

    # README should be the rendered model card with version + history
    readme = (staging / "README.md").read_text()
    assert "LUMINA-TEST" in readme
    assert "v0.1.0" in readme
    assert "## Release History" in readme
    assert "0.082726" in readme  # val_loss row

    assert result.total_parameters > 0
    assert result.repo_id == "argonne/LUMINA-TEST"
    assert result.version == "v0.1.0"


def test_stage_artifacts_extends_existing_history(tmp_path: Path):
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    prior = [
        ReleaseHistoryEntry(
            "v0.1.0", "20251112_193637", "20", "0.083", "All", "a1b2c3d"
        ),
    ]
    existing_readme = (
        "# Some prior card\n\n## Release History\n\n"
        + render_history(prior)
        + "\n\n## Acknowledgements\n"
    )

    uploader = HFUploader(repo_id="org/LUMINA-TEST")
    uploader.stage_artifacts(
        checkpoint_path,
        tmp_path / "release",
        version="v0.2.0",
        existing_readme=existing_readme,
    )
    readme = (tmp_path / "release" / "README.md").read_text()
    assert "v0.2.0" in readme
    assert "v0.1.0" in readme
    # New release listed before the old one
    assert readme.index("v0.2.0") < readme.index("v0.1.0")


def test_stage_artifacts_rejects_bad_checkpoint(tmp_path: Path):
    """Checkpoint without state_dict is rejected by the normalizer."""
    bad_path = tmp_path / "bad.pt"
    torch.save({"not_a_checkpoint": True}, bad_path)

    uploader = HFUploader(repo_id="org/repo")
    with pytest.raises(ValueError, match="model_state_dict"):
        uploader.stage_artifacts(
            bad_path, tmp_path / "staging", version="v0.1.0"
        )


def test_stage_artifacts_rejects_missing_model_kwargs(tmp_path: Path):
    """Lumina-schema checkpoint without model_kwargs cannot be normalized."""
    bad_path = tmp_path / "bad.pt"
    torch.save(
        {
            "model_state_dict": {"x": torch.zeros(1)},
            "config": {"models": {"HGT": {"hidden_channels": 8}}},
        },
        bad_path,
    )
    uploader = HFUploader(repo_id="org/repo")
    with pytest.raises(ValueError, match="model_kwargs"):
        uploader.stage_artifacts(
            bad_path, tmp_path / "staging", version="v0.1.0"
        )


def test_stage_artifacts_missing_license_raises(tmp_path: Path):
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    uploader = HFUploader(
        repo_id="org/repo",
        license_path=tmp_path / "does_not_exist_LICENSE",
    )
    with pytest.raises(FileNotFoundError, match="LICENSE"):
        uploader.stage_artifacts(
            checkpoint_path, tmp_path / "staging", version="v0.1.0"
        )


def test_stage_artifacts_default_license_is_bundled(tmp_path: Path):
    """When license_path=None, the bundled placeholder ships from package data."""
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    uploader = HFUploader(repo_id="org/repo", license_path=None)
    uploader.stage_artifacts(
        checkpoint_path, tmp_path / "staging", version="v0.1.0"
    )
    license_text = (tmp_path / "staging" / "LICENSE").read_text()
    assert "LICENSE — TBD" in license_text
    # Should mention the override flag so maintainers know how to supply
    # the final license once it's approved.
    assert "--license-path" in license_text


# ---------------------------------------------------------------------------
# Real lumina-core checkpoint schema (model_kwargs + config[models])
# ---------------------------------------------------------------------------


def test_normalize_lumina_schema_extracts_config_data():
    """The normalizer builds config_data from model_kwargs + config[models]."""
    _, config_data, state_dict = _build_loaded_modeler()
    raw = _fake_lumina_checkpoint(config_data, state_dict)

    cfg, sd, prov = HFUploader._normalize_checkpoint(raw)

    # Architecture hyperparameters come from model_kwargs (what the
    # trainer actually used), not from the larger config[models][HGT].
    assert cfg["input_channels"] == config_data["input_channels"]
    assert cfg["metadata"] == config_data["metadata"]
    assert "HGT" in cfg["config"]["models"]
    assert cfg["config"]["models"]["HGT"]["hidden_channels"] == 8

    # State dict passes through unchanged
    assert sd is state_dict

    # Provenance: plural case_names; val_loss from best_val_loss
    assert prov["case_names"] == [
        "pglib_opf_case30_ieee",
        "pglib_opf_case57_ieee",
        "pglib_opf_case118_ieee",
    ]
    assert prov["val_loss"] == 0.012242
    assert prov["epoch"] == 27
    assert prov["model_class"] == "lumina.model.opf.hetero_model.HGT"
    assert prov["loss_type"] == "augmented_lagrangian"


def test_normalize_lumina_schema_handles_none_case_names():
    """case_names: None (seen on the 671M ckpt) becomes an empty list."""
    _, config_data, state_dict = _build_loaded_modeler()
    raw = _fake_lumina_checkpoint(config_data, state_dict)
    raw["case_names"] = None
    _, _, prov = HFUploader._normalize_checkpoint(raw)
    assert prov["case_names"] == []


def test_normalize_lumina_schema_model_kwargs_overrides_config():
    """model_kwargs hyperparams win over config[models][arch] (stale fallback)."""
    _, config_data, state_dict = _build_loaded_modeler()
    raw = _fake_lumina_checkpoint(config_data, state_dict)
    # Pretend the training config has a stale hidden_channels value
    raw["config"]["models"]["HGT"]["hidden_channels"] = 9999
    raw["model_kwargs"]["hidden_channels"] = 8  # what was really used
    cfg, _, _ = HFUploader._normalize_checkpoint(raw)
    assert cfg["config"]["models"]["HGT"]["hidden_channels"] == 8


def test_normalize_legacy_schema_passes_through():
    """The 'config_data is already present' fast-path still works."""
    _, config_data, state_dict = _build_loaded_modeler()
    raw = _fake_checkpoint(config_data, state_dict)
    cfg, sd, prov = HFUploader._normalize_checkpoint(raw)
    assert cfg == config_data
    assert sd is state_dict
    # Singular case_name promoted to list
    assert prov["case_names"] == ["pglib_opf_case14_ieee"]
    assert prov["val_loss"] == 0.082726


def test_stage_artifacts_with_lumina_schema_end_to_end(tmp_path: Path):
    """Full stage_artifacts on a real-shaped lumina checkpoint."""
    _, config_data, state_dict = _build_loaded_modeler()
    raw = _fake_lumina_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(raw, checkpoint_path)

    uploader = HFUploader(repo_id="argonne/LUMINA-2M")
    result = uploader.stage_artifacts(
        checkpoint_path, tmp_path / "stage", version="v0.1.0-rc1"
    )

    # All artifacts present
    stage = tmp_path / "stage"
    for f in ("model.pt", "model.safetensors", "config.json",
              "requirements.txt", "README.md", "LICENSE"):
        assert (stage / f).is_file(), f

    # config.json is the normalized config_data
    cfg = json.loads((stage / "config.json").read_text())
    assert cfg["input_channels"] == config_data["input_channels"]
    assert "HGT" in cfg["config"]["models"]
    # Nothing from the training run leaks into config.json
    assert "case_names" not in cfg
    assert "epoch" not in cfg
    assert "best_val_loss" not in cfg

    # README rendered with real provenance
    readme = (stage / "README.md").read_text()
    assert "v0.1.0-rc1" in readme
    assert "0.012242" in readme  # best_val_loss formatted
    # All three case names joined
    assert "pglib_opf_case30_ieee" in readme
    assert "pglib_opf_case57_ieee" in readme
    assert "pglib_opf_case118_ieee" in readme
    assert "Training Epochs**: 27" in readme
    assert "LUMINA-2M" in readme
    # license_other front-matter (placeholder, not Apache-2.0)
    assert "license: other" in readme

    assert result.total_parameters > 0
    assert result.version == "v0.1.0-rc1"


# ---------------------------------------------------------------------------
# HFUploader.upload (mocked Hub)
# ---------------------------------------------------------------------------


def test_upload_invalid_version_raises(tmp_path: Path):
    uploader = HFUploader(repo_id="org/repo")
    with pytest.raises(ValueError, match="Invalid --version"):
        uploader.upload(tmp_path / "x.pt", tmp_path / "stage", version="0.1")


def test_upload_with_mocked_hub(tmp_path: Path):
    """End-to-end upload path with all Hub calls mocked."""
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    fake_commit = MagicMock(
        commit_url="https://huggingface.co/org/repo/commit/deadbee",
        oid="deadbeefcafe0000",
    )

    fake_api_instance = MagicMock()
    fake_api_instance.upload_folder.return_value = fake_commit
    fake_api_instance.create_tag.return_value = None
    fake_api_instance.add_collection_item.return_value = None

    fake_api_cls = MagicMock(return_value=fake_api_instance)
    fake_create_repo = MagicMock()

    fake_hub_module = MagicMock()
    fake_hub_module.HfApi = fake_api_cls
    fake_hub_module.create_repo = fake_create_repo

    with patch.dict("sys.modules", {"huggingface_hub": fake_hub_module}), patch(
        "lumina_inference.release.uploader.fetch_existing_readme",
        return_value=None,
    ):
        uploader = HFUploader(repo_id="org/repo")
        result = uploader.upload(
            checkpoint_path,
            tmp_path / "stage",
            version="v0.2.0",
            collection_slug="org/lumina-collection-xxxx",
        )

    fake_create_repo.assert_called_once()
    fake_api_instance.upload_folder.assert_called_once()
    fake_api_instance.create_tag.assert_called_once()
    tag_kwargs = fake_api_instance.create_tag.call_args.kwargs
    assert tag_kwargs["tag"] == "v0.2.0"
    assert tag_kwargs["revision"] == "main"

    fake_api_instance.add_collection_item.assert_called_once_with(
        collection_slug="org/lumina-collection-xxxx",
        item_id="org/repo",
        item_type="model",
    )

    assert result.version == "v0.2.0"
    assert result.tag_created is True
    assert result.collection_joined is True
    assert result.commit_sha == "deadbee"
    assert result.commit_url == "https://huggingface.co/org/repo/commit/deadbee"


def test_upload_without_collection(tmp_path: Path):
    _, config_data, state_dict = _build_loaded_modeler()
    checkpoint = _fake_checkpoint(config_data, state_dict)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    fake_commit = MagicMock(commit_url="url", oid="abc1234567")
    fake_api_instance = MagicMock()
    fake_api_instance.upload_folder.return_value = fake_commit

    fake_hub_module = MagicMock()
    fake_hub_module.HfApi = MagicMock(return_value=fake_api_instance)
    fake_hub_module.create_repo = MagicMock()

    with patch.dict("sys.modules", {"huggingface_hub": fake_hub_module}), patch(
        "lumina_inference.release.uploader.fetch_existing_readme",
        return_value=None,
    ):
        uploader = HFUploader(repo_id="org/repo")
        result = uploader.upload(
            checkpoint_path, tmp_path / "stage", version="v0.1.0"
        )

    fake_api_instance.add_collection_item.assert_not_called()
    assert result.collection_joined is False
    assert result.tag_created is True
