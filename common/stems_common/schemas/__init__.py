"""JSON Schemas shared by the worker, the API and the tests.

The files here are the parts of the research report's schemas (section 4) that a single
component produces on its own: ``track_features`` (inside schema (b), ``EnsemblePlan``) and
``stem_metrics`` (the ``metrics`` object of schema (c), ``StemQualityAssessment``). The
full schemas are assembled around them by the decision layer.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent


class SchemaError(ValueError):
    """An object does not match its schema. The message lists every violation."""


@cache
def load(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / f"{name}.schema.json"
    if not path.is_file():
        raise KeyError(f"no schema named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def errors(name: str, instance: Any) -> list[str]:
    """Every way ``instance`` breaks the schema ``name``; empty when it is valid."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load(name))
    found = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}" for e in found]


def validate(name: str, instance: Any) -> None:
    problems = errors(name, instance)
    if problems:
        raise SchemaError(f"not a valid {name}: " + "; ".join(problems))
