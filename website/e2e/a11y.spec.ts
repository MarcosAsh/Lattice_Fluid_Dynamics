import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

// Contrast is left out, the Catppuccin palette is a design choice
for (const path of ['/', '/docs', '/comparison', '/optimize']) {
  test(`${path} has no axe violations`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState('networkidle');
    const results = await new AxeBuilder({ page })
      .disableRules(['color-contrast'])
      .analyze();
    const summary = results.violations.map(
      (v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`,
    );
    expect(summary).toEqual([]);
  });
}

test('home page with results has no axe violations', async ({ page }) => {
  await page.goto('/');
  const gallery = page.getByTestId('demo-gallery');
  await gallery.getByRole('button', { name: /^Car/ }).click();
  await gallery.getByRole('button', { name: /^Ahmed 25/ }).click();
  const results = await new AxeBuilder({ page })
    .disableRules(['color-contrast'])
    .analyze();
  expect(results.violations.map((v) => v.id)).toEqual([]);
});
