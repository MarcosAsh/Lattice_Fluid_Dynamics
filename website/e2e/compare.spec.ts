import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('two demo runs can be compared side by side', async ({ page }) => {
  await page.goto('/');
  const gallery = page.getByTestId('demo-gallery');
  await gallery.getByRole('button', { name: /^Car/ }).click();
  await gallery.getByRole('button', { name: /^Ahmed 25/ }).click();

  await page.getByRole('checkbox', { name: 'Compare Car at 1.0 m/s' }).check();
  await expect(page.getByTestId('run-comparison')).toHaveCount(0);
  await page
    .getByRole('checkbox', { name: 'Compare Ahmed 25° at 1.0 m/s' })
    .check();

  const panel = page.getByTestId('run-comparison');
  await expect(panel).toBeVisible();
  await expect(panel.getByText('Car at 1.0 m/s')).toBeVisible();
  await expect(panel.getByText('Ahmed 25° at 1.0 m/s')).toBeVisible();
  const cdRow = panel.getByRole('row', { name: /Drag coefficient/ });
  await expect(cdRow).toContainText('0.7772');
  await expect(cdRow).toContainText('2.3002');
  await expect(cdRow).toContainText('+1.5230 (+196.0%)');
  await expect(panel.locator('video')).toHaveCount(2);

  const axe = await new AxeBuilder({ page })
    .include('[data-testid="run-comparison"]')
    .disableRules(['color-contrast'])
    .analyze();
  expect(axe.violations.map((v) => v.id)).toEqual([]);

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export CSV' }).click(),
  ]);
  expect(download.suggestedFilename()).toBe(
    'lattice_compare_car_w1.0_vs_ahmed25_w1.0.csv',
  );
  const csv = await readFile(await download.path(), 'utf8');
  expect(csv).toContain('step,cd_a,cl_a,cd_b,cl_b');
  expect(csv).toContain('0,1.946000,0.243000,3.779000,0.296000');

  await panel.getByRole('button', { name: 'Close comparison' }).click();
  await expect(panel).toHaveCount(0);
  await expect(
    page.getByRole('checkbox', { name: 'Compare Car at 1.0 m/s' }),
  ).not.toBeChecked();
});

test('a third run cannot be added to a comparison', async ({ page }) => {
  await page.goto('/');
  const gallery = page.getByTestId('demo-gallery');
  for (const name of [/^Car/, /^Ahmed 25/, /^Ahmed 35/]) {
    await gallery.getByRole('button', { name }).click();
  }
  await page.getByRole('checkbox', { name: /^Compare Car/ }).check();
  await page.getByRole('checkbox', { name: /^Compare Ahmed 25/ }).check();
  await expect(
    page.getByRole('checkbox', { name: /^Compare Ahmed 35/ }),
  ).toBeDisabled();
});
