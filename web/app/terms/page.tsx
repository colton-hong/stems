import type { Metadata } from "next";

import { Notice } from "@/components/Notice";

export const metadata: Metadata = { title: "Terms (placeholder)" };

export default function Terms() {
  return (
    <div className="prose">
      <h1>Terms</h1>
      <Notice kind="warning">
        <p>
          <strong>This page is a placeholder.</strong> It is not legal text, nobody with legal
          training has written or read it, and it creates no agreement. Real terms would have to
          be written before this were offered to anyone else.
        </p>
      </Notice>
      <p>Until then, this is what the project is, in plain words:</p>
      <h2>What this is</h2>
      <p>
        A personal, non-commercial project that splits a song into stems using openly available
        models. It is run by one person for their own use.
      </p>
      <h2>What you upload</h2>
      <p>
        Only upload audio you have the right to process. The models used here come with licences
        of their own, several of which allow research or non-commercial use only, so the stems are
        not cleared for commercial use.
      </p>
      <h2>No promises</h2>
      <p>
        There is no guarantee that the service is available, that a separation is good, or that a
        file is kept for any length of time.
      </p>
    </div>
  );
}
