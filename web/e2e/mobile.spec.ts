import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { playerReady, uploadAndWait } from "./helpers";

// A phone held upright, as Chrome emulates one: 375 by 812 CSS pixels, a touch screen and
// a mobile viewport. It is still desktop Chrome's engine underneath, not a phone.
test.use({ viewport: { width: 375, height: 812 }, hasTouch: true, isMobile: true });

/** How many pixels of the page lie beyond the right edge of the screen. */
async function sideways(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

/**
 * Of the controls a finger is meant to press, how many are showing and which are smaller
 * than 24 by 24 CSS pixels (the minimum target size of WCAG 2.2).
 */
async function targets(page: Page): Promise<{ showing: number; small: string[] }> {
  return page.evaluate(() => {
    const showing = Array.from(
      document.querySelectorAll<HTMLElement>("button, select, a.button, summary"),
    ).filter((element) => element.offsetParent !== null);
    const small = showing
      .filter((element) => {
        const box = element.getBoundingClientRect();
        return box.width < 24 || box.height < 24;
      })
      .map((element) => element.getAttribute("aria-label") ?? element.innerText.trim());
    return { showing: showing.length, small };
  });
}

test("on a phone the whole flow works and nothing runs off the side", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeVisible();
  expect(await sideways(page)).toBe(0);
  const upload = await targets(page);
  expect(upload.showing).toBeGreaterThanOrEqual(3); // the button and the two menus, at least
  expect(upload.small).toEqual([]);

  const id = await uploadAndWait(page, { name: "A Rather Long File Name For A Phone Screen.wav" });
  await playerReady(page);
  expect(await sideways(page)).toBe(0);
  const job = await targets(page);
  expect(job.showing).toBeGreaterThanOrEqual(8); // play, and solo, mute and download per stem
  expect(job.small).toEqual([]);
  // The page is taller than the screen but no wider: it scrolls one way only.
  expect(
    await page.evaluate(() => document.documentElement.scrollHeight > window.innerHeight),
  ).toBe(true);
  // Each stem's controls sit under its name instead of beside it, inside the screen.
  for (const row of await page.locator("ul.stems > li").all()) {
    const box = await row.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(375);
  }

  // Play by touch, solo one stem, and download it.
  await page.getByRole("button", { name: "Play", exact: true }).tap();
  await expect.poll(() => page.evaluate(() => window.__stemsEngine?.isPlaying)).toBe(true);
  await page.getByRole("button", { name: "Solo Bass" }).tap();
  await expect(page.getByRole("button", { name: "Solo Bass", pressed: true })).toBeVisible();
  await page.getByRole("button", { name: "Pause", exact: true }).tap();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download Bass" }).tap();
  expect((await download).suggestedFilename()).toContain("(bass)");

  await page.goto("/jobs");
  await expect(page.getByRole("list", { name: "Jobs" }).getByRole("listitem")).toHaveCount(1);
  expect(await sideways(page)).toBe(0);
  for (const path of ["/terms", "/privacy", `/jobs/${id}`]) {
    await page.goto(path);
    expect(await sideways(page), path).toBe(0);
  }
});

test("on a phone axe finds nothing wrong on the upload, job and jobs pages", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await uploadAndWait(page);
  await playerReady(page);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.goto("/jobs");
  await expect(page.getByRole("list", { name: "Jobs" })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});
