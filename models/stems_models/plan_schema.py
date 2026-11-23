"""Schema (b), ``EnsemblePlan``, with its model lists taken from the registry.

The research report's schema names placeholder model families. Here each stem's
``primary_model`` and ``fallback_model`` may be ``none`` or any loadable registry model
that is a candidate for that stem, and the de-reverb model may be null or any loadable
de-reverb candidate. The committed file is the generated one; a test fails when it no
longer matches the registry.

    python -m stems_models.plan_schema            # check the committed file
    python -m stems_models.plan_schema --write    # regenerate it
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any

from stems_common import schemas
from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry, load_registry

NAME = "ensemble_plan"
NONE = "none"
PATH = schemas.SCHEMA_DIR / f"{NAME}.schema.json"
# Keys a standalone schema file carries that an embedded copy of it does not.
_FILE_ONLY = ("$schema", "$id", "title", "description")


def model_enums(registry: Registry) -> dict[str, list[str]]:
    """For each stem and for ``dereverb``, the registry ids a plan may name, in registry order."""
    return {
        slot: [m.id for m in registry.candidates_for(slot)] for slot in (*PRODUCT_STEMS, "dereverb")
    }


def _stem_plan(models: list[str]) -> dict[str, Any]:
    allowed = [NONE, *models]
    return {
        "type": "object",
        "required": ["primary_model", "fallback_model", "requested"],
        "additionalProperties": False,
        "properties": {
            "requested": {"type": "boolean"},
            "primary_model": {"type": "string", "enum": allowed},
            "fallback_model": {"type": "string", "enum": allowed},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def ensemble_plan_schema(registry: Registry | None = None) -> dict[str, Any]:
    """The committed schema with every model list rebuilt from ``registry``."""
    enums = model_enums(registry or load_registry())
    schema = copy.deepcopy(schemas.load(NAME))
    features = {k: v for k, v in schemas.load("track_features").items() if k not in _FILE_ONLY}
    schema["properties"]["track_features"] = copy.deepcopy(features)
    schema["properties"]["plan"]["properties"] = {
        stem: {"$ref": f"#/$defs/stemPlan_{stem}"} for stem in PRODUCT_STEMS
    }
    schema["properties"]["dereverb"]["properties"]["model"] = {
        "type": ["string", "null"],
        "enum": [*enums["dereverb"], None],
    }
    schema["$defs"] = {f"stemPlan_{stem}": _stem_plan(enums[stem]) for stem in PRODUCT_STEMS}
    return schema


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def is_current(registry: Registry | None = None) -> bool:
    return PATH.read_text(encoding="utf-8") == render(ensemble_plan_schema(registry))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--write"]:
        PATH.write_text(render(ensemble_plan_schema()), encoding="utf-8")
        schemas.load.cache_clear()
        print(f"wrote {PATH}")
        return 0
    if args:
        print("usage: python -m stems_models.plan_schema [--write]", file=sys.stderr)
        return 2
    if is_current():
        print(f"{PATH.name} matches the registry")
        return 0
    print(
        f"{PATH.name} does not match the registry; run: python -m stems_models.plan_schema --write",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
