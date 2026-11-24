"""The whole pipeline on the fixture: with a mock of Jev, and with no key at all."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.settings import (  # noqa: E402
    Calibrated,
    EnsembleSettings,
    SlotModels,
    load_defaults,
)
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.registry import ModelEntry, Registry  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402
from stems_worker.decision import job  # noqa: E402
from stems_worker.decision.jev import JevClient  # noqa: E402
from stems_worker.decision.job import JobError, JobRequest, providers, run_job  # noqa: E402
from stems_worker.decision.mock import (  # noqa: E402
    RecordingTransport,
    ScriptedTransport,
    mock_client,
    read_exchanges,
)
from stems_worker.fakes import OracleSeparator  # noqa: E402
from stems_worker.listening.audioset import default_stem_map  # noqa: E402

SR = 44100
LSB_24 = 2.0**-23
FOUR = ["vocals", "drums", "bass", "other"]
UNCHECKED = "This stem could not be quality-checked automatically."
EMPTY = (
    "This stem is silent or very quiet compared with the song. The instrument may not be in "
    "the song, or it was not found."
)


def _entry(model_id: str, stems: list[str], slots: list[str]) -> ModelEntry:
    return ModelEntry.model_construct(
        id=model_id,
        name=f"Fake {model_id}",
        architecture="fake",
        stems=stems,
        candidate_for=slots,
        status="available",
        kind="separator",
        loader="msst",
        files=[],
    )


REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        _entry("fake4", FOUR, FOUR),
        _entry("fake4b", FOUR, FOUR),
        _entry("fake6", list(PRODUCT_STEMS), ["guitar", "piano"]),
        _entry("fake6b", list(PRODUCT_STEMS), ["guitar", "piano"]),
        _entry("fake_kara", ["lead_vocals", "backing_vocals"], ["vocal_split"]),
        _entry("fake_kara2", ["lead_vocals", "backing_vocals"], ["vocal_split"]),
        _entry("fake_dry", ["dry"], ["dereverb"]),
        _entry("fake_dry2", ["dry"], ["dereverb"]),
    ],
)


def _slot(first: str, second: str) -> SlotModels:
    return SlotModels(
        single_best=first, two_model_blend=[first, second], full_ensemble=[first, second]
    )


BASE = load_defaults()
# These tests exercise the retry machinery with silent stems, so the switch that sends an
# empty stem round again is on here. As shipped it is off: see the test of that below.
_RETRYING = BASE.gate.model_copy(
    update={"rules": BASE.gate.rules.model_copy(update={"retry_empty": True})}
)
DEFAULTS = BASE.model_copy(
    update={
        "gate": _RETRYING,
        "ensembles": EnsembleSettings(
            provisional=True,
            basis="fake models for a test",
            default_mode="single_best",
            slots={
                **{stem: _slot("fake4", "fake4b") for stem in FOUR},
                "guitar": _slot("fake6", "fake6b"),
                "piano": _slot("fake6", "fake6b"),
                "vocal_split": _slot("fake_kara", "fake_kara2"),
                "dereverb": _slot("fake_dry", "fake_dry2"),
            },
        ),
    }
)
CALIBRATED = DEFAULTS.model_copy(
    update={"listening": BASE.listening.model_copy(update={"calibrated": Calibrated.both(True)})}
)
SETTINGS = DEFAULTS.decision


class Song:
    """A fixture track on disk, and fake models that know its stems.

    ``broken`` names outputs a model returns as silence, to give the gate something to
    catch: ``{"fake4": ("drums",)}``. ``faint`` names outputs it returns 20 dB down, which
    is what leftovers of an instrument that is not in the song look like.
    """

    def __init__(self, tmp_path, broken: dict | None = None, faint: dict | None = None) -> None:
        track = generate_track(seed=41, duration_seconds=3.0)
        q = lambda x: (np.round(x.astype(np.float64) / LSB_24) * LSB_24).astype(np.float32)  # noqa: E731
        self.stems = {name: q(stem) for name, stem in track.stems.items()}
        self.mix = sum(self.stems.values()).astype(np.float32)
        self.path = tmp_path / "song.wav"
        save_audio(self.path, self.mix, SR, "wav")
        four = {k: self.stems[k] for k in ("vocals", "drums", "bass")}
        four["other"] = self.stems["other"] + self.stems["guitar"] + self.stems["piano"]
        self.lead = q(track.lead_vocals)
        kara = {"lead_vocals": self.lead, "backing_vocals": self.stems["vocals"] - self.lead}
        dry = {"dry": self.stems["vocals"] * np.float32(0.5)}
        self.truth = {
            "fake4": four,
            "fake4b": four,
            "fake6": self.stems,
            "fake6b": self.stems,
            "fake_kara": kara,
            "fake_kara2": kara,
            "fake_dry": dry,
            "fake_dry2": dry,
        }
        self.broken = broken or {}
        for model_id, names in (faint or {}).items():
            self.truth[model_id] = {
                name: stem * np.float32(0.1) if name in names else stem
                for name, stem in self.truth[model_id].items()
            }
        self.tmp = tmp_path

    def factory(self, model_id, stems):
        silence = self.broken.get(model_id, ())
        return [OracleSeparator(self.truth[model_id], model_id=model_id, silence=silence)]

    def run(self, out, request=None, **options):
        request = request or {}
        request.setdefault("cache_dir", self.tmp / "cache")
        request.setdefault("timings_path", self.tmp / "timings.jsonl")
        request.setdefault("device", "cpu")
        options.setdefault("defaults", DEFAULTS)
        return run_job(
            self.path,
            out,
            JobRequest(**request),
            registry=REGISTRY,
            factory=self.factory,
            fingerprint=lambda model_id: f"test-{model_id}",
            rates={},
            **options,
        )

    def written(self, out, name: str, ext: str = "wav"):
        return load_audio(out / f"{name}.{ext}").samples


@pytest.fixture
def song(tmp_path) -> Song:
    return Song(tmp_path)


# --- a stand-in for Jev -------------------------------------------------------------------


def _choice(question: dict, picked: str, confidence: float = 0.9) -> dict:
    options = list(question["criteria"])
    rest = (1 - confidence) / (len(options) - 1)
    return {
        "type": "choice",
        "choice": picked,
        "confidence": confidence,
        "probabilities": {o: (confidence if o == picked else rest) for o in options},
    }


def responder(body: dict) -> dict:
    """Answers as a sensible judge would: what the text names, the first model, and a
    stem is good unless it is silent."""
    state, answers = body["state"], {}
    text = str(state.get("request_text", "")).lower() if isinstance(state, dict) else ""
    silent = isinstance(state, dict) and state.get("metrics", {}).get("silence_fraction", 0) >= 0.98
    for qid, question in body["questions"].items():
        if qid.startswith("stem."):
            answers[qid] = {"type": "noul", "noul": 0.97 if qid[5:] in text else 0.02}
        elif qid == "format":
            answers[qid] = _choice(question, "flac" if "flac" in text else "unspecified")
        elif qid == "quality" and question["type"] == "choice":
            answers[qid] = _choice(question, "unspecified")
        elif qid == "vocal_split":
            answers[qid] = {"type": "noul", "noul": 0.03}
        elif qid.startswith("model.") or qid == "dereverb":
            answers[qid] = _choice(question, next(iter(question["criteria"])))
        elif qid == "ensemble":
            answers[qid] = {"type": "noul", "noul": 0.05}
        elif qid == "quality":
            levels = [0.9, 0.1, 0.0, 0.0, 0.0] if silent else [0.0, 0.0, 0.0, 0.5, 0.5]
            answers[qid] = {
                "type": "score",
                "score": 0.1 if silent else 3.5,
                "confidence": 0.92,
                "legend": {},
                "probabilities": {str(i): p for i, p in enumerate(levels)},
            }
        elif qid == "acceptable":
            answers[qid] = {"type": "noul", "noul": 0.02 if silent else 0.96}
        elif qid == "retry_action":
            answers[qid] = _choice(question, "retry_fallback_model")
        else:
            raise AssertionError(f"unexpected question {qid}")
    return answers


def recording_client(path) -> tuple[JevClient, ScriptedTransport]:
    scripted = ScriptedTransport(responder)
    transport = RecordingTransport(scripted, path, origin="scripted")
    return JevClient(transport, SETTINGS, provider="mock", sleep=lambda s: None), scripted


def decided(result: dict) -> list[tuple]:
    """The decisions of a job, without what legitimately differs between two runs."""
    return [
        (
            d["kind"],
            d["provider"],
            d.get("stem"),
            d["fallback_reason"],
            json.dumps(d["answers"], sort_keys=True),
        )
        for d in result["decisions"]
    ]


# --- acceptance ---------------------------------------------------------------------------

TEXT = "Please give me the vocals and the drums, as FLAC"


def test_the_whole_pipeline_with_the_mock(song, tmp_path) -> None:
    recorded = tmp_path / "recorded.jsonl"
    client, scripted = recording_client(recorded)
    out = tmp_path / "out"
    result = song.run(out, {"text": TEXT}, client=client)

    # Stems.
    assert result["status"] == "ready"
    assert sorted(p.name for p in out.iterdir()) == ["drums.flac", "job.json", "vocals.flac"]
    assert json.loads((out / "job.json").read_text()) == result
    for stem in ("vocals", "drums"):
        assert float(np.abs(song.written(out, stem, "flac") - song.stems[stem]).max()) < 1e-6
        report = result["stems"][stem]
        assert (report["state"], report["warning"], report["retries"]) == ("delivered", None, 0)
        assert report["model_used"] == "fake4"
        assert report["file"] == f"{stem}.flac"
        assert report["attempts"][0]["verdict"]["deliver"] is True
    assert result["warnings"] == {}

    # A complete decision log: one parse, one plan, one gate decision per stem.
    assert [(d["kind"], d.get("stem")) for d in result["decisions"]] == [
        ("parse", None),
        ("plan", None),
        ("gate", "vocals"),
        ("gate", "drums"),
    ]
    assert providers(result) == {"parse": {"mock"}, "plan": {"mock"}, "gate": {"mock"}}
    for entry in result["decisions"]:
        assert entry["model"] == SETTINGS.jev_model
        assert entry["fallback_reason"] is None
        assert entry["input_tokens"] == 0
        assert entry["latency_ms"] >= 0
        assert entry["answers"]
        assert len(entry["request_sha256"]) == 64
    assert result["parsed_request"]["output_format"] == "flac"
    assert result["parsed_request"]["needs_review"] is False
    assert result["plan"]["plan"]["vocals"]["primary_model"] == "fake4"
    assert result["plan"]["plan"]["vocals"]["fallback_model"] == "fake4b"
    assert result["plan"]["plan"]["guitar"]["requested"] is False
    assert result["track_features"]["vocal_presence"] == "unknown"  # no tagger in this test
    assert result["format"] == "flac"
    assert result["job_id"] == result["plan"]["job_id"] == result["parsed_request"]["request_id"]

    # The same job again, answered from the recording alone.
    assert len(scripted.bodies) == 4
    assert [e["origin"] for e in read_exchanges(recorded)] == ["scripted"] * 4
    replayed = mock_client(SETTINGS, recorded)
    again = song.run(tmp_path / "again", {"text": TEXT}, client=replayed, job_id=result["job_id"])
    assert decided(again) == decided(result)
    assert again["stems"] == result["stems"]
    assert again["plan"] == result["plan"]
    assert len(scripted.bodies) == 4  # nobody was asked the second time
    assert replayed._transport.unused() == []


def test_the_whole_pipeline_with_no_key(song, tmp_path) -> None:
    out = tmp_path / "out"
    result = song.run(out, {"text": TEXT}, client=None)

    assert result["status"] == "ready"
    assert sorted(p.name for p in out.iterdir()) == ["drums.flac", "job.json", "vocals.flac"]
    for stem in ("vocals", "drums"):
        assert float(np.abs(song.written(out, stem, "flac") - song.stems[stem]).max()) < 1e-6
    # All three decisions were made by the fallbacks, and the log says why.
    assert providers(result) == {"parse": {"fallback"}, "plan": {"fallback"}, "gate": {"fallback"}}
    assert [(d["kind"], d.get("stem")) for d in result["decisions"]] == [
        ("parse", None),
        ("plan", None),
        ("gate", "vocals"),
        ("gate", "drums"),
    ]
    for entry in result["decisions"]:
        assert entry["provider"] == "fallback"
        assert entry["fallback_reason"] == "no_key"
        assert entry["asked"] is None
        assert entry["model"] is None
        assert entry["answers"] is None
    assert result["parsed_request"]["stems"]["vocals"]["wanted"] is True
    assert result["plan"]["confidence"] == 1.0
    # Nothing listened to the stems, so the gate says so instead of vouching for them.
    for stem in ("vocals", "drums"):
        report = result["stems"][stem]
        assert report["state"] == "delivered_with_warning"
        assert report["warning"] == UNCHECKED
        assert report["retries"] == 0
        assert report["attempts"][0]["assessment"]["reason_code"] == "low_confidence"
    assert result["warnings"] == {"vocals": UNCHECKED, "drums": UNCHECKED}
    assert all(run["stage"] != "listening" for run in result["models"])  # nothing listened
    assert result["listening"] == {
        "tagger": None,
        "calibrated": {"presence": False, "bleed": False},
    }


# --- retries ------------------------------------------------------------------------------


def test_a_silent_stem_is_retried_with_the_fallback_model_no_key(tmp_path) -> None:
    song = Song(tmp_path, broken={"fake4": ("drums",)})
    out = tmp_path / "out"
    result = song.run(out, {"stems": ("drums", "bass")}, client=None)
    drums = result["stems"]["drums"]
    assert [a["model_used"] for a in drums["attempts"]] == ["fake4", "fake4b"]
    first, second = drums["attempts"]
    assert first["assessment"]["reason_code"] == "mostly_silent"
    assert first["verdict"]["action"] == "retry_fallback_model"
    assert first["verdict"]["retry_model"] == "fake4b"
    # The retry cannot be verified without a tagger, but it is better than silence.
    assert second["verdict"]["action"] == "deliver_with_warning"
    assert (drums["state"], drums["delivered_attempt"], drums["retries"]) == (
        "delivered_with_warning",
        1,
        1,
    )
    assert drums["model_used"] == "fake4b"
    assert float(np.abs(song.written(out, "drums") - song.stems["drums"]).max()) < 1e-6
    assert result["stems"]["bass"]["retries"] == 0
    assert [(d["kind"], d.get("stem"), d.get("attempt")) for d in result["decisions"]] == [
        ("parse", None, None),
        ("plan", None, None),
        ("gate", "drums", 0),
        ("gate", "drums", 1),
        ("gate", "bass", 0),
    ]
    assert result["decisions"][0]["provider"] == "selection"
    assert ("fake4b", ["drums"]) in [(m["model_id"], m["outputs"]) for m in result["models"]]


def test_a_retry_that_passes_is_delivered_without_a_warning(tmp_path) -> None:
    song = Song(tmp_path, broken={"fake4": ("drums",)})
    client, _ = recording_client(tmp_path / "recorded.jsonl")
    result = song.run(tmp_path / "out", {"text": "just the drums"}, client=client)
    drums = result["stems"]["drums"]
    assert [a["verdict"]["action"] for a in drums["attempts"]] == [
        "retry_fallback_model",
        "deliver",
    ]
    assert (drums["state"], drums["warning"], drums["model_used"]) == ("delivered", None, "fake4b")
    assert providers(result)["gate"] == {"mock"}
    assert float(np.abs(song.written(tmp_path / "out", "drums") - song.stems["drums"]).max()) < 1e-6


def test_when_every_attempt_fails_the_cap_holds_and_the_best_is_delivered(tmp_path) -> None:
    song = Song(tmp_path, broken={"fake4": ("drums",), "fake4b": ("drums",)})
    result = song.run(tmp_path / "out", {"stems": ("drums",)}, client=None)
    drums = result["stems"]["drums"]
    # First attempt, the fallback model, then the blend of both: two retries, no more.
    assert [a["model_used"] for a in drums["attempts"]] == ["fake4", "fake4b", "fake4+fake4b"]
    assert drums["retries"] == DEFAULTS.gate.max_retries_per_stem == 2
    assert drums["state"] == "delivered_with_warning"
    assert drums["delivered_attempt"] == 0  # all equally bad: the earliest
    assert drums["warning"] == EMPTY


def test_a_stem_far_below_the_mix_is_retried_and_then_delivered_with_its_own_note(tmp_path) -> None:
    # Neither model finds drums: each returns them 20 dB down. Not silence, so the
    # silence rule says nothing; the level against the mix does.
    song = Song(tmp_path, faint={"fake4": ("drums",), "fake4b": ("drums",)})
    result = song.run(tmp_path / "out", {"stems": ("drums", "bass")}, client=None)
    drums, bass = result["stems"]["drums"], result["stems"]["bass"]
    assert [a["model_used"] for a in drums["attempts"]] == ["fake4", "fake4b", "fake4+fake4b"]
    for attempt in drums["attempts"]:
        assert attempt["metrics"]["silence_fraction"] < 0.5
        assert attempt["assessment"]["reason_code"] == "mostly_silent"
        assert attempt["assessment"]["confidence"] == 1.0
    assert (drums["state"], drums["warning"]) == ("delivered_with_warning", EMPTY)
    # A stem at an ordinary level is not touched by the rule.
    assert bass["warning"] == UNCHECKED
    assert bass["retries"] == 0

    # One model does find them: its output is delivered, with the usual note.
    song = Song(tmp_path / "second", faint={"fake4": ("drums",)})
    result = song.run(tmp_path / "second" / "out", {"stems": ("drums",)}, client=None)
    drums = result["stems"]["drums"]
    assert [a["model_used"] for a in drums["attempts"]] == ["fake4", "fake4b"]
    assert (drums["delivered_attempt"], drums["warning"]) == (1, UNCHECKED)


def test_as_shipped_an_empty_stem_is_delivered_at_once_and_no_other_model_runs(tmp_path) -> None:
    assert BASE.gate.rules.retry_empty is False
    shipped = DEFAULTS.model_copy(update={"gate": BASE.gate})
    # Drums come back silent from one model and 20 dB down from the other.
    for label, song in (
        ("silent", Song(tmp_path / "a", broken={"fake4": ("drums",)})),
        ("faint", Song(tmp_path / "b", faint={"fake4": ("drums",)})),
    ):
        (song.tmp).mkdir(parents=True, exist_ok=True)
        song.path = song.tmp / "song.wav"
        save_audio(song.path, song.mix, SR, "wav")
        result = song.run(
            song.tmp / "out", {"stems": ("drums", "bass")}, client=None, defaults=shipped
        )
        drums = result["stems"]["drums"]
        assert [a["model_used"] for a in drums["attempts"]] == ["fake4"], label
        assert drums["attempts"][0]["assessment"]["reason_code"] == "mostly_silent"
        assert drums["attempts"][0]["verdict"]["action"] == "deliver_with_warning"
        assert (drums["state"], drums["retries"], drums["warning"]) == (
            "delivered_with_warning",
            0,
            EMPTY,
        )
        # The fallback model was never run.
        assert {run["model_id"] for run in result["models"]} == {"fake4"}


class OracleTagger:
    """Hears a stem's AudioSet class in audio that is that stem of the fixture track."""

    name = "oracle"
    window_seconds = 10.0

    def __init__(self, song: Song) -> None:
        self.song = song
        stem_map = default_stem_map()
        self.classes = len(stem_map.labels.names)
        self.index = {stem: stem_map.stems[stem][0] for stem in PRODUCT_STEMS}

    def class_probabilities(self, audio, sample_rate):
        row = np.zeros((1, self.classes))
        for stem, truth in self.song.stems.items():
            if audio.shape == truth.shape and float(np.abs(audio - truth).max()) < 1e-4:
                row[0, self.index[stem]] = 0.95
        return row


def test_with_a_calibrated_tagger_the_fallback_gate_delivers_clean_stems(song, tmp_path) -> None:
    result = song.run(
        tmp_path / "out",
        {"stems": ("vocals", "bass")},
        client=None,
        tagger=OracleTagger(song),
        defaults=CALIBRATED,
    )
    assert providers(result)["gate"] == {"fallback"}
    for stem in ("vocals", "bass"):
        report = result["stems"][stem]
        assert (report["state"], report["warning"]) == ("delivered", None)
        assessment = report["attempts"][0]["assessment"]
        assert assessment == {
            "score": 3.0,
            "score_label": "good",
            "pass_probability": 1.0,
            "confidence": 1.0,
            "reason_code": "clean",
        }
        assert report["attempts"][0]["metrics"]["sdr_db"] is None
        assert report["attempts"][0]["metrics"]["bleed_estimate"] == {
            "suspected_source": "none",
            "level": "none",
        }
    # What the tagger cost is in the job's timings: the mix once, then each stem's one attempt.
    heard = [run for run in result["models"] if run["stage"] == "listening"]
    assert [run["outputs"] for run in heard] == [["mix"], ["vocals"], ["bass"]]
    for run in heard:
        assert (run["model_id"], run["device"], run["cached"]) == ("oracle", "cpu", False)
        assert run["audio_seconds"] == 3.0
        assert run["wall_seconds"] >= 0
    assert [run["stage"] for run in result["models"]][-3:] == ["listening"] * 3
    assert result["listening"] == {
        "tagger": "oracle",
        "calibrated": {"presence": True, "bleed": True},
    }
    assert result["warnings"] == {}


# --- everything else ----------------------------------------------------------------------


def test_a_request_with_no_stem_stops_for_review(song, tmp_path) -> None:
    out = tmp_path / "out"
    result = song.run(out, {"text": "make it sound better"}, client=None)
    assert result["status"] == "needs_review"
    assert result["parsed_request"]["review_reason"] == "no_stem_detected"
    assert result["stems"] == {}
    assert [d["kind"] for d in result["decisions"]] == ["parse"]
    assert sorted(p.name for p in out.iterdir()) == ["job.json"]  # no stems were made


def test_split_and_dereverb_extras(song, tmp_path) -> None:
    out = tmp_path / "out"
    result = song.run(
        out, {"stems": ("vocals",), "split_vocals": True, "dereverb": True}, client=None
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "backing_vocals.wav",
        "job.json",
        "lead_vocals.wav",
        "vocals.wav",
        "vocals_dry.wav",
    ]
    assert float(np.abs(song.written(out, "lead_vocals") - song.lead).max()) < 1e-6
    dry = song.stems["vocals"] * np.float32(0.5)
    assert float(np.abs(song.written(out, "vocals_dry") - dry).max()) < 1e-6
    assert result["plan"]["dereverb"] == {"apply": True, "model": "fake_dry"}
    assert result["extras"]["vocals_dry"] == {
        "stage": "dereverb",
        "model_ids": ["fake_dry"],
        "source": "fake_dry on the vocals stem",
        "file": "vocals_dry.wav",
    }
    assert set(result["extras"]) == {"lead_vocals", "backing_vocals", "vocals_dry"}
    stages = [m["stage"] for m in result["models"]]
    assert stages == ["separation", "vocal_split", "dereverb"]


def test_all_six_stems_by_checkbox(song, tmp_path) -> None:
    out = tmp_path / "out"
    result = song.run(out, {"stems": PRODUCT_STEMS, "fmt": "wav"}, client=None)
    assert set(result["stems"]) == set(PRODUCT_STEMS)
    for stem in PRODUCT_STEMS:
        assert float(np.abs(song.written(out, stem) - song.stems[stem]).max()) < 1e-6, stem
    assert result["residual"]["ok"] is True
    assert [d["kind"] for d in result["decisions"]] == ["parse", "plan", *["gate"] * 6]
    assert result["decisions"][0]["provider"] == "selection"
    assert result["input"]["duration_seconds"] == 3.0


def test_a_job_needs_exactly_one_kind_of_request(song, tmp_path, monkeypatch) -> None:
    for bad in ({}, {"text": "drums", "stems": ("drums",)}):
        with pytest.raises(JobError, match="either a request text or a list of stems"):
            JobRequest(**bad)
    with pytest.raises(ValueError, match="unknown output format"):
        song.run(tmp_path / "out", {"stems": ("drums",), "fmt": "ogg"}, client=None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("stems_common.env.DOTENV_PATH", tmp_path / "absent.env")
    missing = str(tmp_path / "nope.wav")
    assert job.main([missing, "--stems", "drums", "--no-tagger", "--out", str(tmp_path / "o")]) == 1
