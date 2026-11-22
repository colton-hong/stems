"""Build the right adapter for a registry id.

Adapters are imported only when asked for: each pulls in a different model library, and a
process that runs one family should not have to load them all.
"""

from __future__ import annotations

from collections.abc import Sequence

from stems_models.registry import Registry, load_registry
from stems_worker.separator import Separator


class UnsupportedModelError(ValueError):
    """The registry entry cannot be run as a separator."""


def build_separators(
    model_id: str, stems: Sequence[str] = (), *, registry: Registry | None = None
) -> list[Separator]:
    """Separators that together emit ``stems`` for ``model_id``.

    Usually one. A bag with one model per stem returns only what the stems need: one
    Demucs separator holding just those models, or one MDX separator per stem (their chunk
    lengths differ). Files are downloaded and verified on the way.
    """
    registry = registry or load_registry()
    entry = registry.get(model_id)
    if entry.kind != "separator" or not entry.is_loadable:
        raise UnsupportedModelError(
            f"{model_id} is not a loadable separator (kind {entry.kind}, status {entry.status})"
        )
    per_stem = [f.stem for f in entry.weights if f.stem is not None]
    wanted = [s for s in stems if s in per_stem] if per_stem else []

    if entry.loader == "msst":
        from stems_worker.adapters.msst import MsstSeparator

        return [MsstSeparator.from_registry(model_id, registry=registry)]
    if entry.loader == "demucs":
        from stems_worker.adapters.demucs import DemucsSeparator

        return [DemucsSeparator.from_registry(model_id, stems=wanted or None, registry=registry)]
    if entry.loader == "uvr_mdx_onnx":
        from stems_worker.adapters.mdx import MdxSeparator

        if not per_stem:
            return [MdxSeparator.from_registry(model_id, registry=registry)]
        return [
            MdxSeparator.from_registry(model_id, stem=stem, registry=registry)
            for stem in (wanted or per_stem)
        ]
    if entry.loader == "uvr_vr":
        from stems_worker.adapters.vr import VrSeparator

        return [VrSeparator.from_registry(model_id, registry=registry)]
    raise UnsupportedModelError(f"{model_id}: no adapter for loader {entry.loader!r}")
