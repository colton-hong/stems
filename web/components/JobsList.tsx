"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ConfirmButton } from "@/components/ConfirmButton";
import { Notice } from "@/components/Notice";
import { ApiError, deleteEverything, listJobs, type Job } from "@/lib/api";
import { formatRemaining } from "@/lib/format";
import { statusLabel } from "@/lib/stages";

export function JobsList() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cleared, setCleared] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let current = true;
    listJobs()
      .then((loaded) => {
        if (current) {
          setJobs(loaded);
          setError(null);
        }
      })
      .catch((caught: unknown) => {
        if (current) {
          setError(caught instanceof ApiError ? caught.message : "The jobs could not be loaded.");
        }
      });
    return () => {
      current = false;
    };
  }, [attempt]);

  if (error) {
    return (
      <Notice kind="error">
        <p>{error}</p>
        <p>
          <button type="button" onClick={() => setAttempt((n) => n + 1)}>
            Try again
          </button>
        </p>
      </Notice>
    );
  }
  if (jobs === null) {
    return <p role="status">Loading…</p>;
  }

  return (
    <>
      {cleared && (
        <Notice kind="success">
          <p>Everything from this browser has been deleted from the server.</p>
        </Notice>
      )}
      {jobs.length === 0 ? (
        <p data-testid="no-jobs">
          This browser has no jobs. <Link href="/">Upload a song</Link>
        </p>
      ) : (
        <>
          <section className="card">
            <ul className="joblist" aria-label="Jobs">
              {jobs.map((job) => (
                <li key={job.id}>
                  <span>
                    <Link href={`/jobs/${job.id}`}>{job.name ?? "Untitled upload"}</Link>{" "}
                    <span className={`badge badge--${job.status}`}>{statusLabel(job.status)}</span>
                  </span>
                  <span className="hint">
                    Deleted in {formatRemaining(job.seconds_until_expiry)}
                  </span>
                </li>
              ))}
            </ul>
          </section>
          <section className="card" aria-labelledby="forget-title">
            <h2 id="forget-title">Delete everything</h2>
            <p>
              Removes every upload and stem this browser has on the server, and the cookie that
              ties them to it.
            </p>
            <ConfirmButton
              label="Delete everything"
              question="Delete every job from this browser? This cannot be undone."
              confirmLabel="Yes, delete everything"
              onConfirm={async () => {
                try {
                  await deleteEverything();
                  setCleared(true);
                  setAttempt((n) => n + 1);
                } catch (caught) {
                  setError(
                    caught instanceof ApiError ? caught.message : "Nothing could be deleted.",
                  );
                }
              }}
            />
          </section>
        </>
      )}
    </>
  );
}
