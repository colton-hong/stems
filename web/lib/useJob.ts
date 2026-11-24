"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, eventsUrl, getJob, type Job, type JobEvent } from "@/lib/api";

const POLL_MS = 2000;

export interface JobState {
  job: Job | null;
  events: JobEvent[];
  error: ApiError | null;
  /** Fetch the job again, for example after deleting it. */
  reload: () => void;
}

function inProgress(job: Job | null): boolean {
  return job !== null && (job.status === "queued" || job.status === "running");
}

function asApiError(caught: unknown): ApiError {
  return caught instanceof ApiError ? caught : new ApiError(0, "Something went wrong in the page.");
}

/**
 * One job, kept current. While the job is in progress the page listens to the API's
 * server-sent events; if that stream cannot be kept open it asks again every two
 * seconds instead, so the page still gets there.
 */
export function useJob(id: string): JobState {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [generation, setGeneration] = useState(0);
  const lastEvent = useRef(0);

  const reload = useCallback(() => setGeneration((n) => n + 1), []);

  useEffect(() => {
    let current = true;
    let source: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const load = async (): Promise<Job | null> => {
      try {
        const loaded = await getJob(id);
        if (current) {
          setJob(loaded);
          setError(null);
        }
        return loaded;
      } catch (caught) {
        if (current) {
          setError(asApiError(caught));
        }
        return null;
      }
    };

    const poll = () => {
      timer = setTimeout(async () => {
        const loaded = await load();
        if (current && (loaded === null || inProgress(loaded))) {
          poll();
        }
      }, POLL_MS);
    };

    const listen = () => {
      source = new EventSource(eventsUrl(id), { withCredentials: true });
      source.addEventListener("job", (message) => {
        if (current) {
          setJob(JSON.parse((message as MessageEvent<string>).data) as Job);
          setError(null);
        }
      });
      source.addEventListener("progress", (message) => {
        const event = JSON.parse((message as MessageEvent<string>).data) as JobEvent;
        if (current && event.id > lastEvent.current) {
          lastEvent.current = event.id;
          setEvents((before) => [...before, event]);
        }
      });
      source.addEventListener("end", () => {
        source?.close();
        void load(); // the final state, with stems and warnings
      });
      source.onerror = () => {
        // The stream broke. Stop the browser's own reconnecting and ask plainly instead.
        source?.close();
        if (current) {
          poll();
        }
      };
    };

    void load().then((loaded) => {
      if (current && inProgress(loaded)) {
        listen();
      }
    });

    return () => {
      current = false;
      source?.close();
      if (timer !== null) {
        clearTimeout(timer);
      }
    };
  }, [id, generation]);

  return { job, events, error, reload };
}
