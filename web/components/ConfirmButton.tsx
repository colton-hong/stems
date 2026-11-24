"use client";

import { useState } from "react";

/**
 * A button for something that cannot be undone: the first click asks, the second does it.
 * Nothing is deleted by one stray click or key press.
 */
export function ConfirmButton({
  label,
  question,
  confirmLabel,
  onConfirm,
  busyLabel = "Deleting…",
}: {
  label: string;
  question: string;
  confirmLabel: string;
  onConfirm: () => Promise<void>;
  busyLabel?: string;
}) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);

  if (!asking) {
    return (
      <button type="button" className="danger" onClick={() => setAsking(true)}>
        {label}
      </button>
    );
  }
  return (
    <span className="row" role="group" aria-label={question}>
      <span>{question}</span>
      <button
        type="button"
        className="danger"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          try {
            await onConfirm();
          } finally {
            setBusy(false);
            setAsking(false);
          }
        }}
      >
        {busy ? busyLabel : confirmLabel}
      </button>
      <button type="button" disabled={busy} onClick={() => setAsking(false)}>
        Cancel
      </button>
    </span>
  );
}
