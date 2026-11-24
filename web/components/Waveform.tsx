"use client";

import { useEffect, useRef } from "react";

import type { Peaks } from "@/lib/api";

/**
 * A stem's waveform, drawn from its peaks, with the part already played in the accent
 * colour. It is a picture: the position slider is the control, so this is hidden from
 * assistive technology. Clicking it seeks, as a convenience for a pointer.
 */
export function Waveform({
  peaks,
  fraction,
  dimmed,
  onSeek,
}: {
  peaks: Peaks | null;
  fraction: number;
  dimmed: boolean;
  onSeek: (fraction: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    if (!element) {
      return;
    }
    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(element.clientWidth * ratio));
      const height = Math.max(1, Math.floor(element.clientHeight * ratio));
      if (element.width !== width || element.height !== height) {
        element.width = width;
        element.height = height;
      }
      const context = element.getContext("2d");
      if (!context) {
        return;
      }
      context.clearRect(0, 0, width, height);
      if (!peaks || peaks.buckets === 0) {
        return;
      }
      const style = getComputedStyle(element);
      const played = style.getPropertyValue("--wave-played").trim() || "#1f5fbf";
      const rest = style.getPropertyValue("--wave").trim() || "#8a94a6";
      const middle = height / 2;
      const split = Math.floor(width * fraction);
      context.globalAlpha = dimmed ? 0.35 : 1;
      for (let x = 0; x < width; x += 1) {
        const bucket = Math.min(peaks.buckets - 1, Math.floor((x / width) * peaks.buckets));
        const low = peaks.min[bucket] ?? 0;
        const high = peaks.max[bucket] ?? 0;
        const top = middle - high * middle;
        const bottom = middle - low * middle;
        context.fillStyle = x < split ? played : rest;
        context.fillRect(x, top, 1, Math.max(1, bottom - top));
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    return () => observer.disconnect();
  }, [peaks, fraction, dimmed]);

  return (
    <div
      className="wave"
      aria-hidden="true"
      onClick={(event) => {
        const box = event.currentTarget.getBoundingClientRect();
        onSeek(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)));
      }}
    >
      <canvas ref={canvas} />
      <div className="wave-head" style={{ left: `${fraction * 100}%` }} />
    </div>
  );
}
