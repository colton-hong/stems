"use client";

import { useEffect, useState } from "react";

/** The current time, refreshed every `everyMs`. Null until the page is in the browser. */
export function useNow(everyMs: number): number | null {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const timer = setInterval(tick, everyMs);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [everyMs]);
  return now;
}
