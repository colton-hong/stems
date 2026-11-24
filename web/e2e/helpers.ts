import { expect, type Page } from "@playwright/test";

const SAMPLE_RATE = 44100;

/**
 * A real WAV file: two tones, 16-bit stereo. 100 Hz lands in the band the stand-in
 * models call bass and 6 kHz in the band they call vocals, so both stems have sound.
 * `extra` adds further tones, each a frequency and an amplitude.
 */
export function toneWav(seconds = 3, extra: Array<[hz: number, amplitude: number]> = []): Buffer {
  const frames = Math.floor(seconds * SAMPLE_RATE);
  const data = Buffer.alloc(frames * 4);
  for (let i = 0; i < frames; i += 1) {
    const t = i / SAMPLE_RATE;
    let sample = 0.3 * Math.sin(2 * Math.PI * 100 * t) + 0.2 * Math.sin(2 * Math.PI * 6000 * t);
    for (const [hz, amplitude] of extra) {
      sample += amplitude * Math.sin(2 * Math.PI * hz * t);
    }
    const value = Math.round(sample * 32767);
    data.writeInt16LE(value, i * 4);
    data.writeInt16LE(value, i * 4 + 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(2, 22); // channels
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * 4, 28);
  header.writeUInt16LE(4, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

export async function chooseFile(page: Page, name: string, buffer: Buffer): Promise<void> {
  await page.locator('input[type="file"]').setInputFiles({ name, mimeType: "audio/wav", buffer });
}

/** Upload a song through the page and wait until its job page says Ready. */
export async function uploadAndWait(
  page: Page,
  options: { name?: string; stems?: string[]; format?: string } = {},
): Promise<string> {
  const { name = "Test Tone.wav", stems = ["Vocals", "Bass"], format } = options;
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Split a song into stems" })).toBeVisible();
  await chooseFile(page, name, toneWav());
  for (const stem of stems) {
    await page.getByRole("checkbox", { name: stem, exact: true }).check();
  }
  if (format) {
    await page.getByLabel("File format").selectOption(format);
  }
  await page.getByRole("button", { name: "Upload and separate" }).click();
  await page.waitForURL(/\/jobs\/[0-9a-f-]{36}$/);
  await expect(page.getByTestId("status")).toHaveText("Ready", { timeout: 120_000 });
  return page.url().split("/").pop() ?? "";
}

export async function playerReady(page: Page): Promise<void> {
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeEnabled({
    timeout: 60_000,
  });
}
