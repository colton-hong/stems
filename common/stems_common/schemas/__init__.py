"""JSON Schemas shared by the worker, the API and the tests.

The three contracts of the research report's section 4, as files: ``parsed_stem_request``
(schema (a)), ``ensemble_plan`` (schema (b), with its model lists generated from the
registry by ``stems_models.plan_schema``) and ``stem_quality_assessment`` (schema (c)).
Two parts of them also stand alone, because one component produces each on its own:
``track_features`` (inside (b)) and ``stem_metrics`` (the ``metrics`` object of (c)).

Formats (``uuid``) are checked, not just annotated.
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


def errors_in(schema: dict[str, Any], instance: Any) -> list[str]:
    """Every way ``instance`` breaks ``schema``; empty when it is valid."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    found = sorted(validator.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}" for e in found]


def errors(name: str, instance: Any) -> list[str]:
    """Every way ``instance`` breaks the schema ``name``; empty when it is valid."""
    return errors_in(load(name), instance)


def validate(name: str, instance: Any) -> None:
    problems = errors(name, instance)
    if problems:
        raise SchemaError(f"not a valid {name}: " + "; ".join(problems))
