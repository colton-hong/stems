"use client";

import { useEffect, useRef, useState } from "react";

import { Notice } from "@/components/Notice";
import { Waveform } from "@/components/Waveform";
import { ApiError, getLinks, getPeaks, type JobStem, type Peaks } from "@/lib/api";
import { formatBytes, formatClock, stemLabel } from "@/lib/format";
import { StemEngine, type ContextLike, type StemState } from "@/lib/player";

/** Tracks up to this long load their audio at once; longer ones wait to be asked. */
const AUTO_LOAD_SECONDS = 600;

type Loading =
  | { state: "idle" }
  | { state: "loading"; done: number; total: number }
  | { state: "ready" }
  | { state: "failed"; message: string };

declare global {
  interface Window {
    /** The player's engine, for the browser tests to read. Not an API. */
    __stemsEngine?: StemEngine;
  }
}

/**
 * Fetch and decode every stem and hand them to one engine, in display order and only
 * once all are decoded, so that every stem is there from the first press of Play.
 */
async function loadEngine(
  jobId: string,
  names: string[],
  onProgress: (done: number) => void,
): Promise<{ context: AudioContext; made: StemEngine }> {
  const links = await getLinks(jobId);
  const context = new AudioContext();
  try {
    let done = 0;
    const buffers = await Promise.all(
      names.map(async (name) => {
        const link = links.stems.find((entry) => entry.name === name);
        if (!link) {
          throw new Error(`no link for ${name}`);
        }
        const response = await fetch(link.url);
        if (!response.ok) {
          throw new Error(`the audio for ${name} could not be fetched (${response.status})`);
        }
        const buffer = await context.decodeAudioData(await response.arrayBuffer());
        done += 1;
        onProgress(done);
        return [name, buffer] as const;
      }),
    );
    const made = new StemEngine(context as unknown as ContextLike);
    for (const [name, buffer] of buffers) {
      made.addStem(name, buffer);
    }
    return { context, made };
  } catch (caught) {
    void context.close();
    throw caught;
  }
}

function extensionLabel(stem: JobStem): string {
  return (stem.format ?? "").replace("_", " ").toUpperCase();
}

export function StemPlayer({
  jobId,
  stems,
  durationSeconds,
}: {
  jobId: string;
  stems: JobStem[];
  durationSeconds: number | null;
}) {
  const engine = useRef<StemEngine | null>(null);
  const audio = useRef<AudioContext | null>(null);
  const frame = useRef<number | null>(null);

  const stemCount = stems.length;
  const automatic = durationSeconds !== null && durationSeconds <= AUTO_LOAD_SECONDS;
  const [loading, setLoading] = useState<Loading>(() =>
    automatic ? { state: "loading", done: 0, total: stemCount } : { state: "idle" },
  );
  const [requested, setRequested] = useState(automatic);
  const [peaks, setPeaks] = useState<Record<string, Peaks>>({});
  const [states, setStates] = useState<StemState[]>([]);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(durationSeconds ?? 0);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const names = stems.map((stem) => stem.name).join(",");

  // Waveforms come from the peaks files: small, and available before any audio is.
  useEffect(() => {
    let current = true;
    for (const name of names.split(",")) {
      getPeaks(jobId, name)
        .then((loaded) => {
          if (current) {
            setPeaks((before) => ({ ...before, [name]: loaded }));
          }
        })
        .catch(() => {
          // A missing waveform is not worth an error: the stem still plays.
        });
    }
    return () => {
      current = false;
    };
  }, [jobId, names]);

  // Loading talks to the network and the audio hardware, and reports back as it goes.
  useEffect(() => {
    if (!requested) {
      return;
    }
    let current = true;
    const wanted = names.split(",");
    loadEngine(jobId, wanted, (done) => {
      if (current) {
        setLoading({ state: "loading", done, total: wanted.length });
      }
    })
      .then(({ context, made }) => {
        if (!current) {
          made.dispose();
          void context.close();
          return;
        }
        made.onEnded = () => {
          setPlaying(false);
          setPosition(made.position);
        };
        audio.current = context;
        engine.current = made;
        window.__stemsEngine = made;
        setStates(made.stemStates);
        setDuration(made.duration);
        setLoading({ state: "ready" });
      })
      .catch((caught: unknown) => {
        if (current) {
          const message =
            caught instanceof ApiError
              ? caught.message
              : "The audio could not be loaded into the player. The downloads below still work.";
          setLoading({ state: "failed", message });
        }
      });
    return () => {
      current = false;
      if (frame.current !== null) {
        cancelAnimationFrame(frame.current);
      }
      engine.current?.dispose();
      engine.current = null;
      delete window.__stemsEngine;
      void audio.current?.close();
      audio.current = null;
    };
  }, [requested, jobId, names]);

  // While playing, follow the audio clock.
  useEffect(() => {
    if (!playing) {
      return;
    }
    const tick = () => {
      if (engine.current) {
        setPosition(engine.current.position);
      }
      frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => {
      if (frame.current !== null) {
        cancelAnimationFrame(frame.current);
      }
    };
  }, [playing]);

  const ready = loading.state === "ready";
  const fraction = duration > 0 ? Math.min(1, position / duration) : 0;
  const anySolo = states.some((stem) => stem.solo);

  async function togglePlay() {
    const current = engine.current;
    if (!current) {
      return;
    }
    if (current.isPlaying) {
      current.pause();
      setPlaying(false);
      setPosition(current.position);
    } else {
      await current.play();
      setPlaying(true);
    }
  }

  async function seek(seconds: number) {
    const current = engine.current;
    if (!current) {
      return;
    }
    await current.seek(seconds);
    setPosition(current.position);
  }

  function setStem(name: string, change: "solo" | "mute") {
    const current = engine.current;
    if (!current) {
      return;
    }
    const before = current.stemStates.find((stem) => stem.name === name);
    if (change === "solo") {
      current.setSolo(name, !before?.solo);
    } else {
      current.setMuted(name, !before?.muted);
    }
    setStates(current.stemStates);
  }

  async function download(target: string) {
    // A fresh link for every click: links are short-lived on purpose.
    setDownloadError(null);
    try {
      const links = await getLinks(jobId);
      const url =
        target === "zip" ? links.zip?.url : links.stems.find((link) => link.name === target)?.url;
      if (!url) {
        throw new ApiError(404, "That file is no longer available.");
      }
      window.location.assign(url);
    } catch (caught) {
      setDownloadError(
        caught instanceof ApiError ? caught.message : "The download could not be started.",
      );
    }
  }

  return (
    <section className="card" aria-labelledby="player-title">
      <h2 id="player-title">Listen</h2>

      {loading.state === "idle" && (
        <Notice kind="info">
          <p>
            This is a long track, so its audio is not loaded into the player until you ask.{" "}
            <button
              type="button"
              onClick={() => {
                setLoading({ state: "loading", done: 0, total: stemCount });
                setRequested(true);
              }}
            >
              Load the player
            </button>
          </p>
        </Notice>
      )}
      {loading.state === "loading" && (
        <p role="status">
          Loading audio… {loading.done} of {loading.total} stems
        </p>
      )}
      {loading.state === "failed" && (
        <Notice kind="error">
          <p>{loading.message}</p>
        </Notice>
      )}

      <div className="transport">
        <button
          type="button"
          className="primary"
          onClick={() => void togglePlay()}
          disabled={!ready}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          aria-label="Position"
          min={0}
          max={Math.max(1, Math.floor(duration))}
          step={1}
          value={Math.floor(position)}
          disabled={!ready}
          aria-valuetext={`${formatClock(position)} of ${formatClock(duration)}`}
          onChange={(event) => void seek(Number(event.target.value))}
        />
        <span className="time" aria-hidden="true">
          {formatClock(position)} / {formatClock(duration)}
        </span>
      </div>

      <ul className="stems" aria-label="Stems">
        {stems.map((stem) => {
          const state = states.find((entry) => entry.name === stem.name);
          const silent = Boolean(state?.muted) || (anySolo && !state?.solo);
          const label = stemLabel(stem.name);
          return (
            <li className="stem" key={stem.name} data-stem={stem.name}>
              <div>
                <div className="stem-name">{label}</div>
                <div className="stem-meta">
                  {extensionLabel(stem)}
                  {typeof stem.bytes === "number" ? `, ${formatBytes(stem.bytes)}` : ""}
                </div>
              </div>
              <Waveform
                peaks={peaks[stem.name] ?? null}
                fraction={fraction}
                dimmed={silent}
                onSeek={(part) => void seek(part * duration)}
              />
              <div className="stem-controls">
                <button
                  type="button"
                  className="toggle"
                  aria-pressed={Boolean(state?.solo)}
                  disabled={!ready}
                  onClick={() => setStem(stem.name, "solo")}
                >
                  Solo<span className="visually-hidden"> {label}</span>
                </button>
                <button
                  type="button"
                  className="toggle"
                  aria-pressed={Boolean(state?.muted)}
                  disabled={!ready}
                  onClick={() => setStem(stem.name, "mute")}
                >
                  Mute<span className="visually-hidden"> {label}</span>
                </button>
                <button type="button" onClick={() => void download(stem.name)}>
                  Download<span className="visually-hidden"> {label}</span>
                </button>
              </div>
              {stem.warning && (
                <p className="stem-warning">
                  <strong>Note:</strong> {stem.warning}
                </p>
              )}
            </li>
          );
        })}
      </ul>

      {downloadError && (
        <Notice kind="error">
          <p>{downloadError}</p>
        </Notice>
      )}
      <p style={{ marginTop: "1rem" }}>
        <button type="button" onClick={() => void download("zip")}>
          Download all stems (zip)
        </button>
      </p>
    </section>
  );
}
