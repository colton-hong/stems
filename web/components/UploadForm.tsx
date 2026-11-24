"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState, type DragEvent, type FormEvent } from "react";

import { Notice } from "@/components/Notice";
import {
  ApiError,
  createJob,
  getConfig,
  parseRequest,
  STEMS,
  type Config,
  type OutputFormat,
  type ParsedRequest,
  type QualityMode,
  type StemName,
} from "@/lib/api";
import {
  fileProblem,
  formatBytes,
  formatLabel,
  formatRemaining,
  qualityLabel,
  stemLabel,
} from "@/lib/format";

type Parse =
  | { state: "idle" }
  | { state: "asking" }
  | { state: "done"; parsed: ParsedRequest }
  | { state: "failed"; message: string };

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429 && error.retryAfter !== null) {
      return `${error.message} (${formatRemaining(error.retryAfter)}.)`;
    }
    return error.message;
  }
  return "Something went wrong in the page. Reload it and try again.";
}

export function UploadForm() {
  const router = useRouter();
  const ids = useId();
  const fileInput = useRef<HTMLInputElement>(null);

  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [chosen, setChosen] = useState<Set<StemName>>(new Set());
  const [splitVocals, setSplitVocals] = useState(false);
  const [dereverb, setDereverb] = useState(false);
  const [format, setFormat] = useState<OutputFormat | "">("");
  const [quality, setQuality] = useState<QualityMode>("unspecified");

  const [text, setText] = useState("");
  const [parse, setParse] = useState<Parse>({ state: "idle" });

  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    getConfig()
      .then((loaded) => {
        if (current) {
          setConfig(loaded);
          setConfigError(null);
        }
      })
      .catch((caught: unknown) => {
        if (current) {
          setConfigError(describe(caught));
        }
      });
    return () => {
      current = false;
    };
  }, [attempt]);

  const problem = file && config ? fileProblem(file, config.uploads) : null;
  const uploading = progress !== null;
  const needsStems = chosen.size === 0;
  const unsure = parse.state === "done" && parse.parsed.needs_review;

  function toggle(stem: StemName) {
    setChosen((before) => {
      const after = new Set(before);
      if (after.has(stem)) {
        after.delete(stem);
      } else {
        after.add(stem);
      }
      return after;
    });
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setOver(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      setError(null);
    }
  }

  async function onParse() {
    const wanted = text.trim();
    if (wanted === "") {
      return;
    }
    setParse({ state: "asking" });
    try {
      const { parsed } = await parseRequest(wanted);
      // What was understood goes into the same controls a person would set by hand, so
      // it can be seen and corrected before anything is uploaded.
      setChosen(new Set(STEMS.filter((stem) => parsed.stems[stem]?.wanted)));
      if (parsed.output_format !== "unspecified") {
        setFormat(parsed.output_format);
      }
      if (parsed.quality_mode !== "unspecified") {
        setQuality(parsed.quality_mode);
      }
      if (parsed.vocal_split === "lead_and_backing") {
        setSplitVocals(true);
      }
      setParse({ state: "done", parsed });
    } catch (caught) {
      setParse({ state: "failed", message: describe(caught) });
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || problem || needsStems || uploading) {
      return;
    }
    setError(null);
    setProgress(0);
    try {
      const job = await createJob(
        file,
        {
          stems: STEMS.filter((stem) => chosen.has(stem)),
          splitVocals,
          dereverb,
          format: format === "" ? null : format,
          qualityMode: quality,
        },
        setProgress,
      );
      router.push(`/jobs/${job.id}`);
    } catch (caught) {
      setProgress(null);
      setError(describe(caught));
    }
  }

  if (configError) {
    return (
      <Notice kind="error">
        <p>{configError}</p>
        <p>
          <button type="button" onClick={() => setAttempt((n) => n + 1)}>
            Try again
          </button>
        </p>
      </Notice>
    );
  }
  if (!config) {
    return <p role="status">Loading…</p>;
  }

  const limits = config.uploads;
  const minutes = Math.round(limits.max_duration_seconds / 60);

  return (
    <form onSubmit={onSubmit} noValidate>
      <section className="card" aria-labelledby={`${ids}-file`}>
        <h2 id={`${ids}-file`}>1. Choose a song</h2>
        <label
          className="dropzone"
          data-over={over}
          onDragOver={(event) => {
            event.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInput}
            className="visually-hidden"
            type="file"
            name="file"
            accept={limits.allowed_extensions.map((extension) => `.${extension}`).join(",")}
            onChange={(event) => {
              setFile(event.target.files?.[0] ?? null);
              setError(null);
            }}
          />
          {file ? (
            <>
              <strong>{file.name}</strong>
              <span className="hint">{formatBytes(file.size)}. Choose a different file</span>
            </>
          ) : (
            <>
              <strong>Drop an audio file here, or choose one</strong>
              <span className="hint">
                {limits.allowed_extensions.join(", ")}. Up to {formatBytes(limits.max_bytes)} and{" "}
                {minutes} minutes.
              </span>
            </>
          )}
        </label>
        {problem && (
          <Notice kind="error">
            <p>{problem}</p>
          </Notice>
        )}
      </section>

      <section className="card" aria-labelledby={`${ids}-what`}>
        <h2 id={`${ids}-what`}>2. Say what you want</h2>
        <fieldset>
          <legend>Stems</legend>
          <div className="checks">
            {STEMS.map((stem) => (
              <label className="check" key={stem}>
                <input
                  type="checkbox"
                  checked={chosen.has(stem)}
                  onChange={() => toggle(stem)}
                />
                {stemLabel(stem)}
              </label>
            ))}
          </div>
          <div className="row">
            <button type="button" onClick={() => setChosen(new Set(STEMS))}>
              All six
            </button>
            <button type="button" onClick={() => setChosen(new Set())}>
              None
            </button>
          </div>
        </fieldset>

        <div className="field" style={{ marginTop: "1rem" }}>
          <label htmlFor={`${ids}-text`}>Or describe it in words (optional)</label>
          <textarea
            id={`${ids}-text`}
            value={text}
            maxLength={2000}
            placeholder="For example: just the vocals and the drums, as FLAC"
            onChange={(event) => {
              setText(event.target.value);
              setParse({ state: "idle" });
            }}
          />
          <div className="row">
            <button
              type="button"
              onClick={onParse}
              disabled={text.trim() === "" || parse.state === "asking"}
            >
              {parse.state === "asking" ? "Reading…" : "Fill in from my words"}
            </button>
            <span className="hint">Your choices above are filled in for you to check.</span>
          </div>
        </div>
        {parse.state === "failed" && (
          <Notice kind="error">
            <p>{parse.message}</p>
          </Notice>
        )}
        {parse.state === "done" && unsure && (
          <Notice kind="warning">
            <p>
              <strong>I could not tell for sure what you asked for.</strong>{" "}
              {needsStems
                ? "No stem was recognised in your words. Tick the stems you want above."
                : "Check the ticked stems above and correct them before you upload."}
            </p>
          </Notice>
        )}
        {parse.state === "done" && !unsure && (
          <Notice kind="info">
            <p>
              Understood as:{" "}
              {STEMS.filter((stem) => parse.parsed.stems[stem]?.wanted)
                .map(stemLabel)
                .join(", ")}
              . The ticks above have been set; change them if that is not what you meant.
            </p>
          </Notice>
        )}
      </section>

      <section className="card" aria-labelledby={`${ids}-options`}>
        <h2 id={`${ids}-options`}>3. Options</h2>
        <div className="fields">
          <div className="field">
            <label htmlFor={`${ids}-format`}>File format</label>
            <select
              id={`${ids}-format`}
              value={format}
              onChange={(event) => setFormat(event.target.value as OutputFormat | "")}
            >
              <option value="">Default ({formatLabel(config.default_format)})</option>
              {config.formats.map((option) => (
                <option key={option} value={option}>
                  {formatLabel(option)}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-quality`}>Speed or quality</label>
            <select
              id={`${ids}-quality`}
              value={quality}
              onChange={(event) => setQuality(event.target.value as QualityMode)}
            >
              {config.quality_modes.map((option) => (
                <option key={option} value={option}>
                  {qualityLabel(option)}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="checks">
          <label className="check">
            <input
              type="checkbox"
              checked={splitVocals}
              onChange={(event) => setSplitVocals(event.target.checked)}
            />
            Also split vocals into lead and backing
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={dereverb}
              onChange={(event) => setDereverb(event.target.checked)}
            />
            Also make a de-reverbed vocal
          </label>
        </div>
      </section>

      {error && (
        <Notice kind="error">
          <p>{error}</p>
        </Notice>
      )}
      {uploading && (
        <div className="card">
          <label htmlFor={`${ids}-progress`}>
            {progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : "Checking the file…"}
          </label>
          <progress id={`${ids}-progress`} value={progress} max={1} />
        </div>
      )}

      <div className="row">
        <button
          type="submit"
          className="primary"
          disabled={!file || Boolean(problem) || needsStems || uploading}
        >
          {uploading ? "Uploading…" : "Upload and separate"}
        </button>
        <span className="hint" id={`${ids}-why`}>
          {!file
            ? "Choose a file first."
            : needsStems
              ? "Tick at least one stem."
              : `Only this browser can see the result. It is deleted after ${config.retention_days} days.`}
        </span>
      </div>
    </form>
  );
}
