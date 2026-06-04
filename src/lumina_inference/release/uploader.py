"""Hugging Face Hub uploader for LUMINA checkpoints.

End-to-end workflow:

    1. Load a training checkpoint (``.pt``) with :func:`torch.load`.
    2. Build a :class:`Modeler` from the embedded ``config_data`` and
       ``model_state_dict`` so the live model can be introspected.
    3. Stage release artifacts (``config.json``, ``model.pt``,
       ``model.safetensors``, ``requirements.txt``, ``LICENSE``,
       ``README.md``) in a working directory.
    4. Push the directory to the target Hugging Face repo, create an
       immutable git tag for the release, and (optionally) add the repo
       to a Hugging Face Collection.

This module imports :mod:`huggingface_hub` lazily so users running plain
inference are not forced to install the upload-side extras.

Copyright (c) 2026, UChicago Argonne, LLC
All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from lumina_inference.modeler import Modeler
from lumina_inference.release.model_card import (
    count_parameters,
    format_parameter_count,
    generate_model_card,
)
from lumina_inference.release.release_history import (
    ReleaseHistoryEntry,
    build_history,
    fetch_existing_readme,
)


DEFAULT_REQUIREMENTS = """\
torch>=2.0
torch_geometric>=2.4
huggingface_hub>=0.20
safetensors>=0.4
lumina-inference>=0.1.0
"""


def _write_default_license(dest: Path) -> None:
    """Copy the bundled LICENSE (Apache-2.0 for released weights) to ``dest``.

    Uses ``importlib.resources`` so it works under both source checkouts
    and installed wheels.
    """
    src = (
        resources.files("lumina_inference.release._data").joinpath("LICENSE")
    )
    dest.write_bytes(src.read_bytes())

# Accept v0.2.0, v1.0.0-rc1, etc. Pre-release suffixes are allowed.
_VERSION_RE = re.compile(
    r"^v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$"
)


@dataclass
class UploadResult:
    """Summary of a completed upload."""

    repo_id: str
    staging_dir: Path
    total_parameters: int
    trainable_parameters: int
    version: str
    commit_sha: str = "pending"
    commit_url: Optional[str] = None
    tag_created: bool = False
    collection_joined: bool = False


class HFUploader:
    """Stage and upload a LUMINA checkpoint to the Hugging Face Hub.

    Args:
        repo_id: Target repository, e.g. ``"argonne/LUMINA-2M"``.
        model_name: Public model name used in the model card title.
            Defaults to the final path segment of ``repo_id``.
        device: Device used when constructing the introspection
            ``Modeler``. CPU is almost always appropriate.
        private: If True, create the repo as private.
        token: Hugging Face access token. Falls back to the cached CLI
            token / ``HF_TOKEN`` env var when ``None``.
        license_path: Path to the ``LICENSE`` file copied into each
            release. Defaults to the Apache-2.0 LICENSE bundled with
            the ``lumina_inference.release`` package.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        private: bool = False,
        token: Optional[str] = None,
        license_path: Optional[Path] = None,
    ):
        self.repo_id = repo_id
        self.model_name = model_name or repo_id.split("/")[-1]
        self.device = device or torch.device("cpu")
        self.private = private
        self.token = token
        # When None, stage_artifacts() falls back to the bundled
        # importlib-resources copy (works in installed wheels too).
        self.license_path = Path(license_path) if license_path else None

    # -- staging --------------------------------------------------------

    def _load_checkpoint(self, checkpoint_path: Path) -> dict:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def _build_modeler(self, checkpoint: dict) -> Modeler:
        config_data = checkpoint.get("config_data") or checkpoint.get("config")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get(
            "state_dict"
        )
        if config_data is None or state_dict is None:
            raise ValueError(
                "Checkpoint must contain both 'config_data' (or 'config') "
                "and 'model_state_dict' (or 'state_dict')."
            )
        modeler = Modeler(self.device, verbose=False)
        modeler.load_model(config_data, state_dict)
        return modeler

    def _checkpoint_to_history_entry(
        self,
        checkpoint: Mapping[str, Any],
        version: str,
    ) -> ReleaseHistoryEntry:
        val_loss = checkpoint.get("val_loss")
        val_loss_str = (
            "Unknown" if val_loss is None else f"{float(val_loss):.6f}"
        )
        return ReleaseHistoryEntry(
            version=version,
            date=str(checkpoint.get("timestamp", "Unknown")),
            epochs=str(checkpoint.get("epoch", "Unknown")),
            val_loss=val_loss_str,
            training_case=str(checkpoint.get("case_name", "Unknown")),
            commit="pending",
        )

    def stage_artifacts(
        self,
        checkpoint_path: Path,
        staging_dir: Path,
        *,
        version: str = "unreleased",
        training_data_size: str = "15,000 samples per case",
        existing_readme: Optional[str] = None,
    ) -> UploadResult:
        """Write all release artifacts to ``staging_dir`` and return a summary.

        Does **not** push to the Hub; use :meth:`upload` for that, or
        call this method directly to inspect the artifacts before
        uploading.

        Args:
            checkpoint_path: Path to the ``.pt`` training checkpoint.
            staging_dir: Local directory to populate with release files.
            version: Release version tag (e.g. ``"v0.2.0"``); written
                into the model card. Validated by :meth:`upload`.
            training_data_size: Free-text training data size for the card.
            existing_readme: Optional prior ``README.md`` text used to
                seed the Release History table. When ``None``, the
                table contains only the current release.
        """
        checkpoint_path = Path(checkpoint_path)
        staging_dir = Path(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = self._load_checkpoint(checkpoint_path)
        modeler = self._build_modeler(checkpoint)

        # 1. Copy the raw .pt checkpoint
        shutil.copy2(checkpoint_path, staging_dir / "model.pt")

        # 2. Write config.json (from config_data inside the checkpoint)
        config_data = checkpoint.get("config_data") or checkpoint.get("config")
        config_path = staging_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(_json_safe(config_data), f, indent=2, default=str)

        # 3. Write SafeTensors weights
        from safetensors.torch import save_file

        state_dict = checkpoint.get("model_state_dict") or checkpoint.get(
            "state_dict"
        )
        # Ensure contiguous tensors (safetensors requirement)
        safe_state = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(safe_state, str(staging_dir / "model.safetensors"))

        # 4. requirements.txt
        (staging_dir / "requirements.txt").write_text(DEFAULT_REQUIREMENTS)

        # 5. LICENSE (bundled Apache-2.0 by default; explicit override allowed)
        license_dest = staging_dir / "LICENSE"
        if self.license_path is None:
            _write_default_license(license_dest)
        else:
            if not self.license_path.exists():
                raise FileNotFoundError(
                    f"LICENSE file not found at {self.license_path}; "
                    "pass license_path=None to use the bundled default."
                )
            shutil.copy2(self.license_path, license_dest)

        # 6. Build Release History (prepend the current release)
        new_entry = self._checkpoint_to_history_entry(checkpoint, version)
        release_history = build_history(
            new_entry, existing_readme=existing_readme
        )

        # 7. README.md (the model card)
        card = generate_model_card(
            modeler,
            checkpoint,
            model_name=self.model_name,
            hf_repo_id=self.repo_id,
            model_version=version,
            commit_sha="pending",
            release_history=release_history,
            training_data_size=training_data_size,
        )
        (staging_dir / "README.md").write_text(card)

        total, trainable = count_parameters(modeler.model)
        return UploadResult(
            repo_id=self.repo_id,
            staging_dir=staging_dir,
            total_parameters=total,
            trainable_parameters=trainable,
            version=version,
        )

    # -- upload ---------------------------------------------------------

    @staticmethod
    def _validate_version(version: str) -> None:
        if not _VERSION_RE.match(version):
            raise ValueError(
                f"Invalid --version {version!r}. Expected vMAJOR.MINOR.PATCH "
                "(optionally with a -prerelease suffix), e.g. 'v0.2.0' "
                "or 'v0.2.0-rc1'."
            )

    def upload(
        self,
        checkpoint_path: Path,
        staging_dir: Path,
        version: str,
        *,
        commit_message: Optional[str] = None,
        training_data_size: str = "15,000 samples per case",
        create_repo: bool = True,
        create_tag: bool = True,
        collection_slug: Optional[str] = None,
    ) -> UploadResult:
        """Stage artifacts and push them to the Hugging Face Hub.

        Args:
            checkpoint_path: Path to the ``.pt`` training checkpoint.
            staging_dir: Local directory to stage into before upload.
            version: Release version tag, e.g. ``"v0.2.0"``. Must match
                ``vMAJOR.MINOR.PATCH`` (with optional pre-release
                suffix). Used both in the model card and as the HF git
                tag created after the push.
            commit_message: Optional custom commit message.
            training_data_size: Free-text training data size for the card.
            create_repo: If True, ensure the repo exists (idempotent).
            create_tag: If True, create an immutable HF revision tag
                pointing at the uploaded commit.
            collection_slug: Optional HF Collection slug (e.g.
                ``"argonne/lumina-models-xxxx"``). When provided,
                the repo is added to the Collection (idempotent).
        """
        self._validate_version(version)

        try:
            from huggingface_hub import HfApi, create_repo as hf_create_repo
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for uploads. "
                "Install with: pip install 'lumina-inference[release]'"
            ) from exc

        # Fetch the current README so the new card extends the history
        # table instead of overwriting it.
        existing_readme = fetch_existing_readme(self.repo_id, token=self.token)

        result = self.stage_artifacts(
            checkpoint_path,
            staging_dir,
            version=version,
            training_data_size=training_data_size,
            existing_readme=existing_readme,
        )

        if create_repo:
            hf_create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=self.private,
                exist_ok=True,
            )

        api = HfApi(token=self.token)
        message = (
            commit_message
            or f"Release {self.model_name} {version} "
            f"({format_parameter_count(result.total_parameters)} params)"
        )
        commit = api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(result.staging_dir),
            commit_message=message,
        )
        result.commit_url = getattr(commit, "commit_url", None) or str(commit)
        result.commit_sha = (
            getattr(commit, "oid", None)
            or getattr(commit, "commit_id", None)
            or "unknown"
        )[:7]

        if create_tag:
            try:
                api.create_tag(
                    repo_id=self.repo_id,
                    tag=version,
                    revision="main",
                    tag_message=f"Release {self.model_name} {version}",
                )
                result.tag_created = True
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"[warn] Tag creation for {version} failed: {exc}. "
                    "The upload itself succeeded; tag may already exist."
                )

        if collection_slug:
            result.collection_joined = _add_to_collection(
                api, collection_slug, self.repo_id
            )

        return result


def _add_to_collection(api, collection_slug: str, repo_id: str) -> bool:
    """Idempotently add ``repo_id`` to a Hugging Face Collection."""
    try:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type="model",
        )
        return True
    except Exception as exc:
        # "already exists" responses look different across hub versions;
        # treat any failure as best-effort and surface a warning.
        msg = str(exc).lower()
        if "already" in msg or "exists" in msg:
            return True
        print(
            f"[warn] Adding {repo_id} to collection "
            f"{collection_slug!r} failed: {exc}"
        )
        return False


def _json_safe(obj):
    """Recursively coerce non-JSON-serializable keys/values (e.g. tuples)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# -- CLI entry point -------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumina-upload",
        description="Upload a LUMINA checkpoint to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a LUMINA training checkpoint (.pt).",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Target HF repo id, e.g. 'argonne/LUMINA-2M'.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version tag, e.g. 'v0.2.0'. Required for all "
        "uploads (including --dry-run for accurate card rendering).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Public model name. Defaults to last segment of --repo.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("./hf_release"),
        help="Directory for staged release artifacts.",
    )
    parser.add_argument(
        "--training-data-size",
        default="15,000 samples per case",
        help="Free-text training data size for the model card.",
    )
    parser.add_argument(
        "--license-path",
        type=Path,
        default=None,
        help="Override path to LICENSE (defaults to the bundled "
        "Apache-2.0 LICENSE in lumina_inference.release._data).",
    )
    parser.add_argument(
        "--collection-slug",
        default=None,
        help="Optional HF Collection slug to add the repo to "
        "(e.g. 'argonne/lumina-models-xxxx').",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip creating an HF git tag for the release.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF access token (defaults to cached CLI token / HF_TOKEN).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message for the upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage artifacts to --staging-dir but skip the upload.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    """Console entry point for ``lumina-upload``."""
    args = _build_arg_parser().parse_args(argv)

    HFUploader._validate_version(args.version)

    uploader = HFUploader(
        repo_id=args.repo,
        model_name=args.name,
        private=args.private,
        token=args.token,
        license_path=args.license_path,
    )

    if args.dry_run:
        # Try to fetch existing README so dry-run cards preview the
        # extended history table. Falls back to fresh history on failure.
        existing_readme = fetch_existing_readme(args.repo, token=args.token)
        result = uploader.stage_artifacts(
            args.checkpoint,
            args.staging_dir,
            version=args.version,
            training_data_size=args.training_data_size,
            existing_readme=existing_readme,
        )
        print(
            f"[dry-run] Staged {result.repo_id} {result.version} "
            f"to {result.staging_dir} "
            f"({format_parameter_count(result.total_parameters)} params)"
        )
        return 0

    result = uploader.upload(
        args.checkpoint,
        args.staging_dir,
        version=args.version,
        commit_message=args.commit_message,
        training_data_size=args.training_data_size,
        create_tag=not args.no_tag,
        collection_slug=args.collection_slug,
    )
    print(
        f"Uploaded {result.repo_id} {result.version} "
        f"({format_parameter_count(result.total_parameters)} params)\n"
        f"  staging dir:        {result.staging_dir}\n"
        f"  commit:             {result.commit_url}\n"
        f"  tag created:        {result.tag_created}\n"
        f"  added to collection:{result.collection_joined}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
