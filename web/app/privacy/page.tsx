import type { Metadata } from "next";

import { Notice } from "@/components/Notice";

export const metadata: Metadata = { title: "Privacy (placeholder)" };

export default function Privacy() {
  return (
    <div className="prose">
      <h1>Privacy</h1>
      <Notice kind="warning">
        <p>
          <strong>This page is a placeholder.</strong> It is not a privacy policy and not legal
          text. It describes how the software is built to behave today; a real policy would have to
          be written before this were offered to anyone else.
        </p>
      </Notice>
      <h2>Who can see your files</h2>
      <p>
        There are no accounts. Your browser gets a random cookie with its first upload, and only
        that browser can list, play, download or delete its jobs. Clearing the cookie means losing
        access to them. There is no sharing and no public link.
      </p>
      <h2>What is kept, and for how long</h2>
      <p>
        The file you upload and the stems made from it are deleted automatically after the number
        of days shown on each job, and at once when you press Delete. Download links stop working
        after a few minutes and are made fresh each time you ask.
      </p>
      <h2>What is not done</h2>
      <p>
        Uploads are not fingerprinted or matched against any catalogue, their tags are not read,
        and nothing is imported from a web address.
      </p>
      <h2>What leaves the server</h2>
      <p>
        If the operator has switched on the optional decision service, it is sent measurements of
        the audio (levels, lengths, what kind of sound was detected) and the words of a typed
        request. It is never sent the audio.
      </p>
    </div>
  );
}
