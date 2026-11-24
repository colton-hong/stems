// The stages a job goes through, by the names the page shows (PLAN.md, "Job stages").

export interface StageView {
  key: string;
  label: string;
  state: "done" | "current" | "pending";
}

const ORDER: Array<{ key: string; label: string }> = [
  { key: "uploaded", label: "Uploaded" },
  { key: "queued", label: "Waiting for a worker" },
  { key: "transcoding", label: "Reading the audio" },
  { key: "analysing", label: "Listening to the track" },
  { key: "planning", label: "Choosing models" },
  { key: "separating", label: "Separating stems" },
  { key: "assessing", label: "Checking each stem" },
  { key: "packaging", label: "Packaging downloads" },
  { key: "ready", label: "Ready" },
];

/** Retrying is part of checking: a stem that failed its check is made again. */
function position(stage: string): number {
  const key = stage === "retrying" ? "assessing" : stage;
  return ORDER.findIndex((entry) => entry.key === key);
}

/**
 * The stage list for a job at `stage`. Everything before the current stage is done; a
 * ready job has every stage done. For a job that ended any other way (failed, deleted,
 * expired) nothing is current: the stages it reached are done and the rest are not.
 */
export function stagesFor(status: string, stage: string, reached?: string): StageView[] {
  const ended = status !== "queued" && status !== "running";
  const at = status === "ready" ? ORDER.length : position(ended ? (reached ?? "") : stage);
  return ORDER.map((entry, index) => ({
    ...entry,
    state:
      index < at || status === "ready" ? "done" : index === at && !ended ? "current" : "pending",
  }));
}

export function stageLabel(stage: string): string {
  if (stage === "retrying") {
    return "Trying a stem again";
  }
  return ORDER.find((entry) => entry.key === stage)?.label ?? stage;
}

const STEM_STATES: Record<string, string> = {
  waiting: "waiting",
  separating: "separating",
  assessing: "being checked",
  retrying: "being made again",
  delivered: "done",
  delivered_with_warning: "done, with a note",
};

export function stemStateLabel(state: string): string {
  return STEM_STATES[state] ?? state;
}

const STATUS_LABELS: Record<string, string> = {
  queued: "Waiting",
  running: "Working",
  ready: "Ready",
  failed: "Failed",
  expired: "Expired",
  deleted: "Deleted",
};

export function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status;
}
