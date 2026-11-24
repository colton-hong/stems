"use client";

import Link from "next/link";
import { useState } from "react";

import { ConfirmButton } from "@/components/ConfirmButton";
import { Notice } from "@/components/Notice";
import { StemPlayer } from "@/components/StemPlayer";
import { ApiError, deleteJob, type Job } from "@/lib/api";
import { formatClock, formatLabel, formatRemaining, secondsUntil, stemLabel } from "@/lib/format";
import { stageLabel, stagesFor, statusLabel, stemStateLabel } from "@/lib/stages";
import { useJob } from "@/lib/useJob";
import { useNow } from "@/lib/useNow";

const MARKS = { done: "✓", current: "●", pending: "○" } as const;

function Stages({ job, reached }: { job: Job; reached: string | undefined }) {
  const inProgress = job.status === "queued" || job.status === "running";
  const stages = stagesFor(job.status, job.stage, reached);
  return (
    <section className="card" aria-labelledby="stages-title">
      <h2 id="stages-title">Progress</h2>
      {/* Said aloud when it changes, without stealing focus. */}
      <p className="visually-hidden" role="status" aria-live="polite">
        {inProgress ? `Now: ${stageLabel(job.stage)}` : `Status: ${statusLabel(job.status)}`}
      </p>
      <ol className="stages">
        {stages.map((stage) => (
          <li key={stage.key} data-state={stage.state} data-stage={stage.key}>
            <span className="stage-mark" aria-hidden="true">
              {MARKS[stage.state]}
            </span>
            <span>
              {stage.label}
              <span className="visually-hidden">
                {stage.state === "done"
                  ? " (done)"
                  : stage.state === "current"
                    ? " (in progress)"
                    : " (not started)"}
              </span>
              {stage.state === "current" && job.stage === "retrying" && " (trying a stem again)"}
            </span>
          </li>
        ))}
      </ol>
      {inProgress && job.progress !== null && (
        <progress
          aria-label={`${stageLabel(job.stage)} progress`}
          value={job.progress}
          max={1}
          style={{ marginTop: "0.6rem" }}
        />
      )}
      {inProgress && job.stems.length > 0 && (
        <ul className="stem-states" aria-label="Each stem">
          {job.stems.map((stem) => (
            <li key={stem.name} data-stem={stem.name}>
              {stemLabel(stem.name)}: {stemStateLabel(stem.state)}
              {stem.attempt > 0 ? ` (try ${stem.attempt + 1})` : ""}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Summary({ job }: { job: Job }) {
  const asked = job.request.stems?.map(stemLabel).join(", ") ?? job.request.text ?? "";
  const extras = [
    job.request.split_vocals ? "lead and backing vocals" : null,
    job.request.dereverb ? "de-reverbed vocal" : null,
  ].filter(Boolean);
  return (
    <p className="lede">
      Asked for: {asked}
      {extras.length > 0 ? `, plus ${extras.join(" and ")}` : ""}
      {job.request.format ? `, as ${formatLabel(job.request.format)}` : ""}
      {job.source.duration_seconds !== null
        ? `. Length ${formatClock(job.source.duration_seconds)}.`
        : "."}
    </p>
  );
}

export function JobView({ id }: { id: string }) {
  const { job, events, error, reload } = useJob(id);
  const now = useNow(30_000);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  if (error && !job) {
    return (
      <>
        <h1>{error.notFound ? "Job not found" : "This job could not be loaded"}</h1>
        <Notice kind="error">
          <p>
            {error.notFound
              ? "There is no such job for this browser. A job can only be opened in the browser that uploaded it, and it disappears when it is deleted or its time runs out."
              : error.message}
          </p>
        </Notice>
        <div className="row">
          {!error.notFound && (
            <button type="button" onClick={reload}>
              Try again
            </button>
          )}
          <Link href="/">Upload a song</Link>
        </div>
      </>
    );
  }
  if (!job) {
    return <p role="status">Loading…</p>;
  }

  const gone = job.status === "deleted" || job.status === "expired";
  const warned = job.stems.filter((stem) => stem.warning);
  // The last stage the worker reported, for showing how far a failed job got.
  const reached = [...events].reverse().find((event) => event.stage !== "failed")?.stage;
  const remaining = now === null ? job.seconds_until_expiry : secondsUntil(job.expires_at, now);

  return (
    <>
      <h1>
        {job.name ?? "Job"}{" "}
        <span className={`badge badge--${job.status}`} data-testid="status">
          {statusLabel(job.status)}
        </span>
      </h1>
      {!gone && <Summary job={job} />}

      {error && (
        <Notice kind="warning">
          <p>The connection to the server was lost. Trying again…</p>
        </Notice>
      )}

      {job.status === "deleted" && (
        <Notice kind="success">
          <p>This job was deleted. Its upload and stems are gone from the server.</p>
        </Notice>
      )}
      {job.status === "expired" && (
        <Notice kind="info">
          <p>This job ran out of time and its files were deleted automatically.</p>
        </Notice>
      )}
      {job.status === "failed" && (
        <Notice kind="error">
          <p>
            <strong>This job failed.</strong> {job.error ?? "No reason was given."}
          </p>
        </Notice>
      )}

      {!gone && <Stages job={job} reached={reached} />}

      {job.status === "ready" && warned.length > 0 && (
        <Notice kind="warning">
          <p>
            <strong>
              {warned.length === job.stems.length
                ? "Every stem has a note."
                : `${warned.length} of ${job.stems.length} stems have a note.`}
            </strong>{" "}
            A note means the automatic quality check could not vouch for that stem. Listen before
            you rely on it.
          </p>
        </Notice>
      )}

      {job.status === "ready" && (
        <StemPlayer jobId={job.id} stems={job.stems} durationSeconds={job.source.duration_seconds} />
      )}

      {!gone && (
        <section className="card" aria-labelledby="delete-title">
          <h2 id="delete-title">Deletion</h2>
          <p data-testid="countdown">
            The upload and every stem are deleted automatically in{" "}
            <strong>{formatRemaining(remaining)}</strong>.
          </p>
          {deleteError && (
            <Notice kind="error">
              <p>{deleteError}</p>
            </Notice>
          )}
          <ConfirmButton
            label="Delete now"
            question="Delete the upload and all stems? This cannot be undone."
            confirmLabel="Yes, delete"
            onConfirm={async () => {
              setDeleteError(null);
              try {
                await deleteJob(job.id);
                reload();
              } catch (caught) {
                setDeleteError(
                  caught instanceof ApiError ? caught.message : "The job could not be deleted.",
                );
              }
            }}
          />
        </section>
      )}

      {gone && (
        <p>
          <Link href="/">Upload another song</Link>
        </p>
      )}
    </>
  );
}
