"""Writes tests/fixtures/jev/plan.handwritten.json. Run from the repo root with the venv."""

import json
import pathlib
import sys

from stems_models.registry import load_registry

REGISTRY = load_registry()
FEATURES = {
    "duration_seconds": 200.0,
    "sample_rate_hz": 44100,
    "channels": 2,
    "vocal_presence": "lead_vocals",
    "detected_instruments": ["vocals", "drums", "bass"],
    "genre_hint": "Rock music",
    "reverb_estimate": None,
    "loudness_lufs": -9.5,
}


def choice(slot, picked, confidence, second=None, p=None):
    options = [m.id for m in REGISTRY.candidates_for(slot)]
    assert picked in options
    assert second is None or second in options
    p = confidence if p is None else p
    probabilities = dict.fromkeys(options, 0.0)
    probabilities[picked] = p
    rest = [o for o in options if o != picked]
    if second is not None:
        probabilities[second] = round((1 - p) * 0.75, 4)
        rest.remove(second)
        left = round((1 - p) * 0.25, 4)
    else:
        left = round(1 - p, 4)
    for o in rest:
        probabilities[o] = round(left / len(rest), 6)
    if not rest:
        # Two candidates only: whatever the picked one leaves all goes to the other, so
        # that the probabilities sum to 1 and the reply is one the parser accepts.
        other = second if second is not None else picked
        probabilities[other] = round(probabilities[other] + left, 4)
    return {
        "type": "choice",
        "choice": picked,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def noul(p):
    return {"type": "noul", "noul": p}


CASES = {
    "confident": {
        "stems": ["vocals", "drums"],
        "quality_mode": "unspecified",
        "dereverb": False,
        "answers": {
            "model.vocals": choice(
                "vocals", "bs_roformer_viperx_vocals", 0.86, "mel_roformer_kim_vocals"
            ),
            "model.drums": choice("drums", "htdemucs_ft", 0.82, "scnet_xl_ihf"),
            "ensemble": noul(0.08),
        },
    },
    "unsure_drums": {
        "stems": ["vocals", "drums"],
        "quality_mode": "best_quality",
        "dereverb": False,
        "answers": {
            "model.vocals": choice(
                "vocals", "mel_roformer_kim_vocals", 0.9, "bs_roformer_viperx_vocals"
            ),
            "model.drums": choice("drums", "htdemucs_ft", 0.41, "scnet_xl_ihf", p=0.45),
            "ensemble": noul(0.12),
        },
    },
    "all_unsure": {
        "stems": ["bass"],
        "quality_mode": "balanced",
        "dereverb": False,
        "answers": {
            "model.bass": choice("bass", "scnet_xl", 0.35, "htdemucs_ft", p=0.4),
            "ensemble": noul(0.55),
        },
    },
    "best_quality_blend": {
        "stems": ["vocals", "drums", "bass", "guitar", "piano", "other"],
        "quality_mode": "best_quality",
        "dereverb": False,
        "answers": {
            "model.vocals": choice(
                "vocals", "mel_roformer_kim_vocals", 0.91, "bs_roformer_viperx_vocals"
            ),
            "model.drums": choice("drums", "scnet_xl_ihf", 0.88, "htdemucs_ft"),
            "model.bass": choice("bass", "htdemucs_ft", 0.85, "bs_roformer_sw_6stem"),
            "model.guitar": choice(
                "guitar", "bs_roformer_sw_6stem", 0.9, "mel_roformer_guitar_becruily"
            ),
            "model.piano": choice("piano", "bs_roformer_sw_6stem", 0.93, "htdemucs_6s"),
            "model.other": choice("other", "scnet_xl_ihf", 0.84, "scnet_large"),
            "ensemble": noul(0.93),
        },
    },
    "balanced_blend": {
        "stems": ["drums"],
        "quality_mode": "balanced",
        "dereverb": False,
        "answers": {
            "model.drums": choice("drums", "scnet_xl_ihf", 0.9, "htdemucs_ft"),
            "ensemble": noul(0.9),
        },
    },
    "dereverb": {
        "stems": ["vocals"],
        "quality_mode": "fastest",
        "dereverb": True,
        "answers": {
            "model.vocals": choice(
                "vocals", "mel_roformer_kim_vocals", 0.95, "bs_roformer_viperx_vocals"
            ),
            "dereverb": choice(
                "dereverb", "mel_roformer_dereverb_echo_sucial", 0.9, "mel_roformer_dereverb_anvuew"
            ),
        },
    },
}
out = {
    "about": "Hand-written answers in Jev's wire format, for tests of how answers are mapped "
    "to an EnsemblePlan. They are not recordings of Jev and say nothing about which model "
    "Jev would pick; real recordings come with P4.9. The request body is built by the test "
    "from the current registry and question wording. Regenerate after a registry change "
    "that adds or removes a candidate (the probabilities must cover every option).",
    "origin": "handwritten",
    "track_features": FEATURES,
    "cases": CASES,
}
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(path, len(CASES))
