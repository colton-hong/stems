import type { ReactNode } from "react";

type Kind = "error" | "warning" | "info" | "success";

/**
 * A message box. Errors interrupt a screen reader (`alert`); everything else is read
 * when it is next idle (`status`).
 */
export function Notice({ kind, children }: { kind: Kind; children: ReactNode }) {
  return (
    <div className={`notice notice--${kind}`} role={kind === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}
