// Regenerates the docs tutorial screenshots from a running build
// usage: node scripts/tutorial-screenshots.mjs [base url]
import { chromium } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const base = process.argv[2] ?? 'http://localhost:3000';
const outDir = fileURLToPath(new URL('../public/docs/', import.meta.url));
const jsonPath = new URL(
  '../src/components/docs/tutorial-shots.json',
  import.meta.url,
);

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1280, height: 1000 },
  deviceScaleFactor: 2,
  reducedMotion: 'reduce',
});

// Fake a healthy backend so the shots show the live layout
await page.route('**/api/render', (route) =>
  route.request().method() === 'GET'
    ? route.fulfill({ json: { available: true, status: 'healthy' } })
    : route.abort(),
);

await page.goto(base);
await page
  .getByTestId('demo-gallery')
  .getByRole('button', { name: /^Car/ })
  .click();
await page.getByText('ML Estimate').waitFor();
await page.getByText('Drag Coefficient').waitFor();

async function shot(name, panel, targets) {
  const box = await panel.boundingBox();
  if (!box) throw new Error(`${name} panel not visible`);
  const markers = [];
  // Badge on the top right corner of each target
  for (const [n, target] of targets) {
    const t = await target.boundingBox();
    if (!t) throw new Error(`marker ${n} target not visible`);
    markers.push({
      n,
      x: +(((t.x + t.width - box.x) / box.width) * 100).toFixed(1),
      y: +(((t.y - box.y) / box.height) * 100).toFixed(1),
    });
  }
  await panel.screenshot({ path: `${outDir}tutorial-${name}.png` });
  return {
    src: `/docs/tutorial-${name}.png`,
    width: Math.round(box.width),
    height: Math.round(box.height),
    markers,
  };
}

const controls = page.getByTestId('control-panel');
const results = page.getByTestId('results-panel');
const shots = {
  controls: await shot('controls', controls, [
    [1, controls.getByLabel('Model', { exact: true })],
    [2, controls.getByLabel('Wind Speed', { exact: true })],
    [3, controls.getByRole('button', { name: 'Start Render' })],
  ]),
  results: await shot('results', results, [
    [4, results.getByText('Drag Coefficient').locator('..')],
  ]),
};

await writeFile(jsonPath, `${JSON.stringify(shots, null, 2)}\n`);
await browser.close();
