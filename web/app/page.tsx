import { UploadForm } from "@/components/UploadForm";

export default function Home() {
  return (
    <>
      <h1>Split a song into stems</h1>
      <p className="lede">
        Upload a song and get back its vocals, drums, bass, guitar, piano and the rest as separate
        files you can play together or download.
      </p>
      <UploadForm />
    </>
  );
}
