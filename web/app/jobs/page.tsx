import type { Metadata } from "next";

import { JobsList } from "@/components/JobsList";

export const metadata: Metadata = { title: "My jobs" };

export default function Jobs() {
  return (
    <>
      <h1>My jobs</h1>
      <p className="lede">
        Everything this browser has uploaded and not yet deleted. Nobody else can see this list.
      </p>
      <JobsList />
    </>
  );
}
