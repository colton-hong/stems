"""The stem vocabulary used across the registry, config, pipeline and API.

Two different things are both called "other" in the wild, so they get different names here:

- ``other``: the product stem, i.e. whatever is left once vocals, drums, bass, guitar and
  piano are removed. A 4-stem model's "other" still contains guitar and piano.
- ``instrumental``: everything except vocals, as emitted by 2-stem vocal models.
"""

# The six stems v1 ships, in display order.
PRODUCT_STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")

# Outputs a checkpoint may emit that are not product stems themselves.
AUX_OUTPUTS: tuple[str, ...] = ("instrumental", "lead_vocals", "backing_vocals", "dry")

# A slot is a job a checkpoint can be a benchmark candidate for.
SLOTS: tuple[str, ...] = (
    *PRODUCT_STEMS,
    "vocal_split",
    "dereverb",
    "tagging",
    "listening_test",
)

# Which outputs a checkpoint must emit to be a candidate for a slot (any one of them).
# Slots that are missing here put no requirement on outputs.
SLOT_REQUIRED_OUTPUTS: dict[str, tuple[str, ...]] = {
    **{stem: (stem,) for stem in PRODUCT_STEMS},
    "vocal_split": ("lead_vocals", "backing_vocals"),
    "dereverb": ("dry",),
}
