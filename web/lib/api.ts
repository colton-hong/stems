// The typed client for the Stems API (PLAN.md section 8). Every request carries the
// owner cookie, which the API sets with the first upload: that cookie is the only thing
// that ties this browser to its jobs.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

export const STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"] as const;
export type StemName = (typeof STEMS)[number];
export type OutputFormat = "wav" | "flac" | "mp3_320" | "mp3_192";
export type QualityMode = "best_quality" | "balanced" | "fastest" | "unspecified";
export type JobStatus = "queued" | "running" | "ready" | "failed" | "expired" | "deleted";

export interface Config {
  stems: StemName[];
  extra_stems: string[];
  formats: OutputFormat[];
  default_format: OutputFormat;
  quality_modes: QualityMode[];
  /** The ensemble mode each quality setting gives when the rules plan a job. */
  quality_gives: Partial<Record<QualityMode, string>>;
  /** Who plans a job: "jev" may give some settings a blend whatever quality_gives says. */
  decided_by: "jev" | "rules";
  uploads: { max_bytes: number; max_duration_seconds: number; allowed_extensions: string[] };
  retention_days: number;
  link_ttl_seconds: number;
}

export interface StemQuality {
  score: number | null;
  score_label: string | null;
  confidence: number | null;
  reason_code: string | null;
}

export interface JobStem {
  name: string;
  // waiting | separating | assessing | retrying | delivered | delivered_with_warning
  state: string;
  attempt: number;
  format?: OutputFormat;
  bytes?: number | null;
  warning?: string | null;
  quality?: StemQuality | null;
  model_ids?: string[];
  ensemble_mode?: string | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string;
  name: string | null;
  request: {
    stems: StemName[] | null;
    text: string | null;
    split_vocals: boolean;
    dereverb: boolean;
    format: OutputFormat | null;
    quality_mode: QualityMode;
  };
  created_at: string;
  updated_at: string;
  expires_at: string;
  seconds_until_expiry: number;
  source: {
    bytes: number | null;
    duration_seconds: number | null;
    sample_rate_hz: number | null;
    channels: number | null;
  };
  progress: number | null;
  stems: JobStem[];
  error: string | null;
}

export interface JobEvent {
  id: number;
  at: string;
  stage: string;
  stem: string | null;
  message: string | null;
  progress: number | null;
}

export interface DownloadLink {
  name: string;
  filename: string;
  url: string;
}

export interface Links {
  expires_in: number;
  expires_at: string;
  stems: DownloadLink[];
  zip: { filename: string; url: string } | null;
}

export interface StemChoice {
  wanted: boolean;
  probability: number;
  confidence?: number;
}

export interface ParsedRequest {
  raw_text: string;
  stems: Record<StemName, StemChoice>;
  vocal_split?: "single_vocals" | "lead_and_backing" | "unspecified";
  output_format: OutputFormat | "unspecified";
  quality_mode: QualityMode;
  confidence: number;
  needs_review: boolean;
  review_reason?: string;
}

export interface ParseReply {
  parsed: ParsedRequest;
  provider: string;
  fallback_reason: string | null;
}

export interface Peaks {
  version: number;
  sample_rate_hz: number;
  duration_seconds: number;
  buckets: number;
  min: number[];
  max: number[];
}

export interface UploadOptions {
  stems: StemName[];
  splitVocals: boolean;
  dereverb: boolean;
  format: OutputFormat | null;
  qualityMode: QualityMode;
}

/** A request the API answered with an error, or could not be reached at all. */
export class ApiError extends Error {
  /** HTTP status; 0 when the API could not be reached. */
  readonly status: number;
  /** Seconds to wait, when the API said so (429). */
  readonly retryAfter: number | null;

  constructor(status: number, message: string, retryAfter: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }

  get unreachable(): boolean {
    return this.status === 0;
  }

  get notFound(): boolean {
    return this.status === 404;
  }
}

export const UNREACHABLE = "The Stems server could not be reached. Check that it is running, then try again.";

/** What to show for an error body: the API's own words when it gave any. */
export function messageFrom(status: number, body: unknown): string {
  if (body !== null && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.trim() !== "") {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      // FastAPI's validation errors: a list of {loc, msg}.
      const first = detail[0] as { msg?: unknown; loc?: unknown };
      const where = Array.isArray(first.loc) ? String(first.loc[first.loc.length - 1]) : "";
      const what = typeof first.msg === "string" ? first.msg : "is not valid";
      return where ? `${where}: ${what}` : what;
    }
  }
  if (status === 404) {
    return "Not found.";
  }
  return `The server answered with an error (${status}).`;
}

function retryAfterOf(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { credentials: "include", ...init });
  } catch {
    throw new ApiError(0, UNREACHABLE);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new ApiError(
      response.status,
      messageFrom(response.status, body),
      retryAfterOf(response.headers.get("Retry-After")),
    );
  }
  return body as T;
}

export function getConfig(): Promise<Config> {
  return request<Config>("/api/config");
}

export async function listJobs(): Promise<Job[]> {
  return (await request<{ jobs: Job[] }>("/api/jobs")).jobs;
}

export function getJob(id: string): Promise<Job> {
  return request<Job>(`/api/jobs/${encodeURIComponent(id)}`);
}

export function getLinks(id: string): Promise<Links> {
  return request<Links>(`/api/jobs/${encodeURIComponent(id)}/links`, { method: "POST" });
}

export function getPeaks(id: string, stem: string): Promise<Peaks> {
  return request<Peaks>(
    `/api/jobs/${encodeURIComponent(id)}/stems/${encodeURIComponent(stem)}/peaks`,
  );
}

export function deleteJob(id: string): Promise<void> {
  return request<void>(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function deleteEverything(): Promise<void> {
  return request<void>("/api/me", { method: "DELETE" });
}

export function parseRequest(text: string): Promise<ParseReply> {
  return request<ParseReply>("/api/requests/parse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function eventsUrl(id: string): string {
  return `${API_URL}/api/jobs/${encodeURIComponent(id)}/events`;
}

/**
 * Upload a file as a new job. Uses XMLHttpRequest because fetch cannot report how much
 * of an upload has been sent. `onProgress` gets a fraction from 0 to 1.
 */
export function createJob(
  file: File,
  options: UploadOptions,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<Job> {
  return new Promise<Job>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("stems", options.stems.join(","));
    form.append("split_vocals", String(options.splitVocals));
    form.append("dereverb", String(options.dereverb));
    if (options.format !== null) {
      form.append("format", options.format);
    }
    form.append("quality_mode", options.qualityMode);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/jobs`);
    xhr.withCredentials = true;
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(event.loaded / event.total);
      }
    };
    xhr.onerror = () => reject(new ApiError(0, UNREACHABLE));
    xhr.onabort = () => reject(new ApiError(0, "The upload was cancelled."));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as Job);
      } else {
        reject(
          new ApiError(
            xhr.status,
            messageFrom(xhr.status, xhr.response),
            retryAfterOf(xhr.getResponseHeader("Retry-After")),
          ),
        );
      }
    };
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(form);
  });
}
