"""Typed access to models/registry.yaml.

The registry is hand-edited YAML. This module is the only place that parses it, so every
consumer (pipeline, benchmark, API, download scripts) sees the same validated objects.

Run ``python -m stems_models`` to validate the file and print a summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from stems_common.paths import REGISTRY_PATH
from stems_common.stems import AUX_OUTPUTS, PRODUCT_STEMS, SLOT_REQUIRED_OUTPUTS, SLOTS

SUPPORTED_SCHEMA_VERSION = 1

Kind = Literal["separator", "tagger", "decision_model"]
Status = Literal["available", "unavailable", "hosted"]
Tier = Literal["candidate", "baseline", "extra", "watch"]
Origin = Literal["report_s1", "report_s5", "added"]
FileRole = Literal["weights", "config", "config_index"]
Loader = Literal[
    "msst",  # ZFTurbo Music-Source-Separation-Training model classes + YAML config
    "demucs",  # official demucs package, loaded by bag name / signatures
    "uvr_mdx_onnx",  # UVR MDX-Net ONNX models
    "uvr_vr",  # UVR VR-architecture .pth models
    "beats",
    "laion_clap",
    "panns",
    "hf_transformers",
    "essentia_tf",
    "workers_ai",  # hosted on Cloudflare Workers AI, nothing to download
    "none",  # tracked only; no weights to load
]

_ALLOWED_OUTPUTS = frozenset(PRODUCT_STEMS) | frozenset(AUX_OUTPUTS)


class RegistryError(ValueError):
    """The registry file is missing, malformed or internally inconsistent."""


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: FileRole
    filename: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    # Written by stems_models.check_sources from an HTTP HEAD. None = not verified yet.
    size_bytes: int | None = Field(default=None, ge=0)
    # For bags that ship one weight file per stem (htdemucs_ft, kuielab).
    stem: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelFile:
        if "/" in self.filename or self.filename in {".", ".."}:
            raise ValueError(f"filename must be a bare file name, got {self.filename!r}")
        if self.stem is not None and self.stem not in _ALLOWED_OUTPUTS:
            raise ValueError(f"unknown stem {self.stem!r} on file {self.filename}")
        return self


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    name: str = Field(min_length=1)
    kind: Kind
    architecture: str = Field(min_length=1)
    loader: Loader
    # Outputs this checkpoint emits, in the vocabulary of stems_common.stems.
    stems: list[str]
    # Slots this checkpoint is benchmarked for. Empty = registered but not a candidate.
    candidate_for: list[str]
    tier: Tier
    origin: Origin
    status: Status
    sample_rate_hz: int | None = Field(default=None, gt=0)
    source_page: str = Field(pattern=r"^https://")
    files: list[ModelFile]
    # Loader-specific settings; each loader adapter validates its own keys.
    config: dict[str, Any]
    # Informational only. Never used to select or exclude a model.
    licence_note: str = Field(min_length=1)
    notes: str | None = None
    # Facts that still need checking at implementation time.
    unverified: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ModelEntry:
        unknown = [s for s in self.stems if s not in _ALLOWED_OUTPUTS]
        if unknown:
            raise ValueError(f"{self.id}: unknown stems {unknown}")
        if len(set(self.stems)) != len(self.stems):
            raise ValueError(f"{self.id}: duplicate stems")

        bad_slots = [s for s in self.candidate_for if s not in SLOTS]
        if bad_slots:
            raise ValueError(f"{self.id}: unknown slots in candidate_for {bad_slots}")
        for slot in self.candidate_for:
            needed = SLOT_REQUIRED_OUTPUTS.get(slot)
            if needed and not any(out in self.stems for out in needed):
                raise ValueError(
                    f"{self.id}: candidate for {slot!r} but emits none of {list(needed)}"
                )

        has_weights = any(f.role == "weights" for f in self.files)
        if self.status == "available" and not has_weights:
            raise ValueError(f"{self.id}: status is 'available' but no weights file is listed")
        if self.status != "available" and self.files:
            raise ValueError(f"{self.id}: status {self.status!r} must not list files")
        if self.status == "unavailable" and self.candidate_for:
            raise ValueError(f"{self.id}: unavailable models cannot be candidates")
        if (self.status == "available") != (self.loader not in {"none", "workers_ai"}):
            raise ValueError(f"{self.id}: loader {self.loader!r} does not fit status {self.status}")

        names = [f.filename for f in self.files]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicate filenames {names}")
        for f in self.files:
            if f.stem is not None and f.stem not in self.stems:
                raise ValueError(f"{self.id}: file {f.filename} is for a stem the model lacks")
        config_file = self.config.get("config_file")
        if config_file is not None and config_file not in names:
            raise ValueError(f"{self.id}: config_file {config_file!r} is not among its files")
        return self

    @property
    def is_loadable(self) -> bool:
        """True if the pipeline can download and run this checkpoint locally."""
        return self.status == "available"

    @property
    def other_contains_guitar_piano(self) -> bool:
        """True if this model's "other" is the 4-stem kind (guitar and piano still in it).

        Only a model that separates guitar and piano itself emits the 6-stem "other".
        """
        if "other" not in self.stems:
            return False
        return not {"guitar", "piano"} <= set(self.stems)

    @property
    def weights(self) -> list[ModelFile]:
        return [f for f in self.files if f.role == "weights"]

    @property
    def total_size_bytes(self) -> int | None:
        """Sum of all file sizes, or None if any file is still unverified."""
        sizes = [f.size_bytes for f in self.files]
        if not sizes or any(s is None for s in sizes):
            return None
        return sum(s for s in sizes if s is not None)

    def files_for(self, stem: str | None = None) -> list[ModelFile]:
        """Files needed to produce ``stem`` (all files if the model is not split per stem)."""
        if stem is None:
            return list(self.files)
        return [f for f in self.files if f.stem in (None, stem)]


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _check(self) -> Registry:
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} is not supported "
                f"(expected {SUPPORTED_SCHEMA_VERSION})"
            )
        seen: set[str] = set()
        for m in self.models:
            if m.id in seen:
                raise ValueError(f"duplicate model id {m.id!r}")
            seen.add(m.id)
        # One URL must always mean one filename, so downloads can be cached by URL.
        url_names: dict[str, str] = {}
        for m in self.models:
            for f in m.files:
                if url_names.setdefault(f.url, f.filename) != f.filename:
                    raise ValueError(f"URL {f.url} is registered under two filenames")
        return self

    def ids(self) -> list[str]:
        return [m.id for m in self.models]

    def get(self, model_id: str) -> ModelEntry:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model id {model_id!r}")

    def __contains__(self, model_id: object) -> bool:
        return any(m.id == model_id for m in self.models)

    def loadable(self) -> list[ModelEntry]:
        return [m for m in self.models if m.is_loadable]

    def candidates_for(self, slot: str, *, loadable_only: bool = True) -> list[ModelEntry]:
        """Models benchmarked for ``slot``, in registry order."""
        if slot not in SLOTS:
            raise KeyError(f"unknown slot {slot!r}")
        return [
            m
            for m in self.models
            if slot in m.candidate_for and (m.is_loadable or not loadable_only)
        ]


def load_registry(path: Path | str | None = None) -> Registry:
    """Parse and validate the registry. Raises RegistryError with a readable message."""
    path = Path(path) if path is not None else REGISTRY_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RegistryError(f"registry file not found: {path}") from e
    except yaml.YAMLError as e:
        raise RegistryError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level")
    try:
        return Registry.model_validate(raw)
    except ValidationError as e:
        raise RegistryError(f"{path} failed validation:\n{e}") from e


def _format_size(n: int | None) -> str:
    if n is None:
        return "unverified"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.1f} MB"


def summary_table(registry: Registry) -> str:
    """Markdown table of every entry, for humans and for docs."""
    lines = [
        "| id | outputs | candidate for | tier | status | size |",
        "|---|---|---|---|---|---|",
    ]
    for m in registry.models:
        size = _format_size(m.total_size_bytes) if m.files else "-"
        lines.append(
            f"| `{m.id}` | {', '.join(m.stems) or '-'} | {', '.join(m.candidate_for) or '-'} "
            f"| {m.tier} | {m.status} | {size} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        registry = load_registry(argv[0] if argv else None)
    except RegistryError as e:
        print(e, file=sys.stderr)
        return 1
    print(summary_table(registry))
    loadable = registry.loadable()
    sizes = [m.total_size_bytes for m in loadable]
    known = sum(s for s in sizes if s is not None)
    print(
        f"\n{len(registry.models)} entries, {len(loadable)} loadable; "
        f"{_format_size(known)} across the {sum(s is not None for s in sizes)} "
        f"loadable entries whose sizes are verified."
    )
    return 0
