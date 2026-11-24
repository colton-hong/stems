// Small pure helpers for showing sizes, times and names.

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

/** 75.4 -> "1:15"; 3725 -> "1:02:05". */
export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const two = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${two(m)}:${two(s)}` : `${m}:${two(s)}`;
}

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** How long until something happens, in the two largest units: "6 days 23 hours". */
export function formatRemaining(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) {
    return "less than a minute";
  }
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (days > 0) {
    return hours > 0 ? `${plural(days, "day")} ${plural(hours, "hour")}` : plural(days, "day");
  }
  if (hours > 0) {
    return minutes > 0
      ? `${plural(hours, "hour")} ${plural(minutes, "minute")}`
      : plural(hours, "hour");
  }
  return plural(minutes, "minute");
}

export function secondsUntil(iso: string, now: number = Date.now()): number {
  return Math.max(0, Math.floor((Date.parse(iso) - now) / 1000));
}

const STEM_LABELS: Record<string, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  other: "Other",
  lead_vocals: "Lead vocals",
  backing_vocals: "Backing vocals",
  vocals_dry: "Vocals, de-reverbed",
};

export function stemLabel(name: string): string {
  return STEM_LABELS[name] ?? name.replace(/_/g, " ");
}

const FORMAT_LABELS: Record<string, string> = {
  wav: "WAV (24-bit, lossless)",
  flac: "FLAC (lossless, smaller)",
  mp3_320: "MP3 320 kbps",
  mp3_192: "MP3 192 kbps",
};

export function formatLabel(format: string): string {
  return FORMAT_LABELS[format] ?? format;
}

const QUALITY_LABELS: Record<string, string> = {
  unspecified: "Default",
  fastest: "Fastest",
  balanced: "Balanced",
  best_quality: "Best quality (slowest)",
};

export function qualityLabel(mode: string): string {
  return QUALITY_LABELS[mode] ?? mode;
}

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot < 0 ? "" : filename.slice(dot + 1).toLowerCase();
}

/** Why a file cannot be uploaded, or null if nothing visible is wrong with it. */
export function fileProblem(
  file: { name: string; size: number },
  limits: { max_bytes: number; allowed_extensions: string[] },
): string | null {
  const extension = extensionOf(file.name);
  if (!limits.allowed_extensions.includes(extension)) {
    const allowed = [...limits.allowed_extensions].sort().join(", ");
    return `This file type is not supported. Choose one of: ${allowed}.`;
  }
  if (file.size === 0) {
    return "This file is empty.";
  }
  if (file.size > limits.max_bytes) {
    return `This file is ${formatBytes(file.size)}. The limit is ${formatBytes(limits.max_bytes)}.`;
  }
  return null;
}
