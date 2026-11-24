// The stem player's engine: every stem on one audio clock.
//
// Each stem is decoded into an AudioBuffer. To play, one AudioBufferSourceNode per stem
// is created and all of them are started with the same `when` (a time on the context's
// clock, slightly in the future) and the same `offset` into the audio. The audio thread
// then renders them sample for sample together: there is nothing to drift, because no
// stem has a clock of its own. Pausing stops every source and remembers the position;
// seeking is a pause and a play from the new offset. Solo and mute only move gains.

export interface StemState {
  name: string;
  muted: boolean;
  solo: boolean;
}

/** The gain each stem gets: silent if muted, or if another stem is soloed and it is not. */
export function gainsFor(stems: readonly StemState[]): Record<string, number> {
  const anySolo = stems.some((stem) => stem.solo);
  const gains: Record<string, number> = {};
  for (const stem of stems) {
    gains[stem.name] = stem.muted || (anySolo && !stem.solo) ? 0 : 1;
  }
  return gains;
}

/** Where playback is, given where it started. Never past the end. */
export function positionAt(
  playing: boolean,
  offsetAtStart: number,
  startedAt: number,
  now: number,
  duration: number,
): number {
  if (!playing) {
    return Math.min(Math.max(offsetAtStart, 0), duration);
  }
  // Before `startedAt` the sources are scheduled and not yet sounding.
  return Math.min(Math.max(offsetAtStart + Math.max(0, now - startedAt), 0), duration);
}

export function clampSeek(seconds: number, duration: number): number {
  if (!Number.isFinite(seconds)) {
    return 0;
  }
  return Math.min(Math.max(seconds, 0), Math.max(0, duration));
}

// The part of the Web Audio API the engine uses, so tests can stand in for it.
export interface ClockLike {
  readonly currentTime: number;
}
export interface ParamLike {
  value: number;
  setTargetAtTime?(value: number, startTime: number, timeConstant: number): unknown;
}
export interface GainLike {
  readonly gain: ParamLike;
  connect(destination: unknown): unknown;
}
export interface SourceLike {
  buffer: unknown;
  connect(destination: unknown): unknown;
  start(when: number, offset: number): void;
  stop(): void;
  disconnect(): void;
  onended: (() => void) | null;
}
export interface ContextLike extends ClockLike {
  readonly destination: unknown;
  readonly state?: string;
  createGain(): GainLike;
  createBufferSource(): SourceLike;
  resume?(): Promise<void>;
}
export interface BufferLike {
  readonly duration: number;
}

export interface Scheduled {
  name: string;
  when: number;
  offset: number;
}

/** Seconds between asking to play and the first sample: time to set every source up. */
export const START_DELAY = 0.06;
const GAIN_SMOOTHING = 0.015;

export class StemEngine {
  private readonly context: ContextLike;
  private readonly buffers = new Map<string, BufferLike>();
  private readonly gains = new Map<string, GainLike>();
  private sources: SourceLike[] = [];
  private states: StemState[] = [];
  private playing = false;
  private offset = 0;
  private startedAt = 0;
  /** Every start() the engine has issued, newest last: what the tests look at. */
  readonly schedule: Scheduled[] = [];
  onEnded: (() => void) | null = null;

  constructor(context: ContextLike) {
    this.context = context;
  }

  addStem(name: string, buffer: BufferLike): void {
    const gain = this.context.createGain();
    gain.connect(this.context.destination);
    this.buffers.set(name, buffer);
    this.gains.set(name, gain);
    this.states.push({ name, muted: false, solo: false });
    this.applyGains(false);
  }

  get names(): string[] {
    return this.states.map((stem) => stem.name);
  }

  get stemStates(): StemState[] {
    return this.states.map((stem) => ({ ...stem }));
  }

  /** The length of the longest stem. They are all the length of the upload. */
  get duration(): number {
    let longest = 0;
    for (const buffer of this.buffers.values()) {
      longest = Math.max(longest, buffer.duration);
    }
    return longest;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get position(): number {
    return positionAt(
      this.playing,
      this.offset,
      this.startedAt,
      this.context.currentTime,
      this.duration,
    );
  }

  async play(): Promise<void> {
    if (this.playing || this.buffers.size === 0) {
      return;
    }
    if (this.context.state === "suspended" && this.context.resume) {
      await this.context.resume();
    }
    if (this.offset >= this.duration) {
      this.offset = 0;
    }
    // One start time and one offset for every stem: this is the synchronisation.
    const when = this.context.currentTime + START_DELAY;
    const offset = this.offset;
    const created: SourceLike[] = [];
    for (const [name, buffer] of this.buffers) {
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.gains.get(name));
      created.push(source);
    }
    created.forEach((source, index) => {
      source.start(when, offset);
      this.schedule.push({ name: this.names[index] ?? "", when, offset });
    });
    const first = created[0];
    if (first) {
      first.onended = () => {
        if (this.playing && this.sources === created) {
          this.finish();
        }
      };
    }
    this.sources = created;
    this.startedAt = when;
    this.playing = true;
  }

  pause(): void {
    if (!this.playing) {
      return;
    }
    this.offset = this.position;
    this.stopSources();
    this.playing = false;
  }

  /** Jump to `seconds`. Keeps playing if it was playing. */
  async seek(seconds: number): Promise<void> {
    const target = clampSeek(seconds, this.duration);
    const wasPlaying = this.playing;
    if (wasPlaying) {
      this.stopSources();
      this.playing = false;
    }
    this.offset = target;
    if (wasPlaying) {
      await this.play();
    }
  }

  setMuted(name: string, muted: boolean): void {
    this.update(name, (stem) => ({ ...stem, muted }));
  }

  setSolo(name: string, solo: boolean): void {
    this.update(name, (stem) => ({ ...stem, solo }));
  }

  /** Stop everything and forget the audio. */
  dispose(): void {
    this.stopSources();
    this.playing = false;
    this.buffers.clear();
    this.gains.clear();
    this.states = [];
  }

  private update(name: string, change: (stem: StemState) => StemState): void {
    this.states = this.states.map((stem) => (stem.name === name ? change(stem) : stem));
    this.applyGains(true);
  }

  private applyGains(smooth: boolean): void {
    const gains = gainsFor(this.states);
    for (const [name, node] of this.gains) {
      const value = gains[name] ?? 1;
      if (smooth && node.gain.setTargetAtTime) {
        // A short ramp, so that muting does not click.
        node.gain.setTargetAtTime(value, this.context.currentTime, GAIN_SMOOTHING);
      } else {
        node.gain.value = value;
      }
    }
  }

  private stopSources(): void {
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Already stopped: nothing to do.
      }
      source.disconnect();
    }
    this.sources = [];
  }

  private finish(): void {
    this.stopSources();
    this.playing = false;
    this.offset = this.duration;
    this.onEnded?.();
  }
}
