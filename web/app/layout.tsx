import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Stems", template: "%s · Stems" },
  description: "Split a song into stems. Personal, non-commercial project.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip" href="#main">
          Skip to the content
        </a>
        <header className="site-header">
          <Link className="site-name" href="/">
            Stems
          </Link>
          <nav aria-label="Main">
            <Link href="/">New upload</Link>
            <Link href="/jobs">My jobs</Link>
          </nav>
        </header>
        <main className="page" id="main">
          {children}
        </main>
        <footer className="site-footer">
          <span>A personal, non-commercial project.</span>
          <Link href="/terms">Terms</Link>
          <Link href="/privacy">Privacy</Link>
        </footer>
      </body>
    </html>
  );
}
