import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { playerReady, uploadAndWait } from "./helpers";

async function focusedName(page: Page): Promise<string> {
  return page.evaluate(() => {
    const element = document.activeElement as HTMLElement | null;
    return element?.getAttribute("aria-label") ?? element?.innerText?.trim() ?? "";
  });
}

/** Press Tab until the focused control's name matches, as a keyboard user would. */
async function tabTo(page: Page, name: RegExp): Promise<void> {
  for (let presses = 0; presses < 40; presses += 1) {
    await page.keyboard.press("Tab");
    if (name.test(await focusedName(page))) {
      return;
    }
  }
  throw new Error(`no control named ${name} was reached with Tab`);
}

test("the player can be worked with the keyboard alone", async ({ page }) => {
  await uploadAndWait(page);
  await playerReady(page);
  // From here on no mouse: only Tab, Space, Enter and the arrow keys.
  await page.locator("body").focus();
  await tabTo(page, /^Play$/);
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeFocused();
  expect(await page.evaluate(() => window.__stemsEngine?.isPlaying)).toBe(true);

  await page.keyboard.press("Tab");
  const slider = page.getByRole("slider", { name: "Position" });
  await expect(slider).toBeFocused();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await expect(slider).toHaveAttribute("aria-valuetext", /^0:0[2-3] of 0:03$/);
  await page.keyboard.press("Home");

  await tabTo(page, /^Solo\s+Vocals$/);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Solo Vocals", pressed: true })).toBeFocused();
  await tabTo(page, /^Mute\s+Vocals$/);
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Mute Vocals", pressed: true })).toBeFocused();
  expect(await page.evaluate(() => window.__stemsEngine?.stemStates[0])).toEqual({
    name: "vocals",
    muted: true,
    solo: true,
  });

  // Back to the transport and pause, still by keyboard.
  for (let presses = 0; presses < 40; presses += 1) {
    await page.keyboard.press("Shift+Tab");
    if ((await focusedName(page)) === "Pause") {
      break;
    }
  }
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeFocused();
  expect(await page.evaluate(() => window.__stemsEngine?.isPlaying)).toBe(false);
});

test("every player control has a role, a name and a state", async ({ page }) => {
  await uploadAndWait(page);
  await playerReady(page);
  await expect(page.getByRole("region", { name: "Listen" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeEnabled();
  const slider = page.getByRole("slider", { name: "Position" });
  await expect(slider).toHaveAttribute("aria-valuetext", "0:00 of 0:03");
  const stems = page.getByRole("list", { name: "Stems" }).getByRole("listitem");
  await expect(stems).toHaveCount(2);
  for (const stem of ["Vocals", "Bass"]) {
    await expect(page.getByRole("button", { name: `Solo ${stem}`, pressed: false })).toBeEnabled();
    await expect(page.getByRole("button", { name: `Mute ${stem}`, pressed: false })).toBeEnabled();
    await expect(page.getByRole("button", { name: `Download ${stem}` })).toBeEnabled();
  }
  // The waveform is a picture, hidden from assistive technology; nothing else is.
  await expect(page.locator(".wave")).toHaveCount(2);
  for (const wave of await page.locator(".wave").all()) {
    await expect(wave).toHaveAttribute("aria-hidden", "true");
  }
  // Progress is announced politely, and the stage list says each stage's state in words.
  await expect(page.locator('[aria-live="polite"]')).toHaveText("Status: Ready");
  await expect(page.locator("ol.stages > li").first()).toContainText("(done)");
});

test("axe finds nothing wrong on the upload page, the job page and the jobs page", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

  const id = await uploadAndWait(page);
  await playerReady(page);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

  await page.goto("/jobs");
  await expect(page.getByRole("list", { name: "Jobs" })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

  await page.goto(`/jobs/${id}`);
  await page.getByRole("button", { name: "Delete now" }).click();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});
