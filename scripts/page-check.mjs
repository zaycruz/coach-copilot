#!/usr/bin/env node
// Browser check of the landing page: desktop and 390 px phone width. Fails on console errors, horizontal scroll,
// an empty call picker or a missing headline. Does not press Play (that would make live Clef calls).
// Usage: node scripts/page-check.mjs [baseUrl]   (needs playwright: npm i -D playwright, or NODE_PATH to one)
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";
const BASE = (process.argv[2] || "https://coach.raava.dev").replace(/\/$/, "");
const browser = await chromium.launch();
const out = { at: new Date().toISOString(), base: BASE, views: [] };
for (const [name, w, h] of [["desktop", 1280, 900], ["phone", 390, 844]]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.querySelectorAll("#callPick option").length > 0);
  const hscroll = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  const calls = await page.locator("#callPick option").count();
  const headline = (await page.locator(".proof").innerText()).replace(/\s+/g, " ");
  const headlineOk = headline.includes("0.75") && headline.includes("0.70") && headline.includes("0.69");
  await page.screenshot({ path: `runs/page-${name}.png`, fullPage: name === "desktop" });
  out.views.push({ name, width: w, errors, horizontalScroll: hscroll, calls, headline, ok: !errors.length && !hscroll && calls > 0 && headlineOk });
  await page.close();
}
await browser.close();
writeFileSync("runs/page-check-latest.json", JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.views));
process.exit(out.views.every((v) => v.ok) ? 0 : 1);
