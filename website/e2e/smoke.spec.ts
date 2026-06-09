import { test, expect } from '@playwright/test';

// Loads the home page and checks the core UI mounts without an uncaught
// exception. A render-time crash (like the localStorage parse error that
// shipped once) throws before these elements appear, so this would catch it.
test('home page renders the control panel and demo gallery without crashing', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));

  await page.goto('/');

  // Control panel mounted.
  await expect(
    page.getByRole('heading', { name: 'Simulation Parameters' }),
  ).toBeVisible();

  // Demo gallery rendered with at least one example.
  const gallery = page.getByTestId('demo-gallery');
  await expect(gallery).toBeVisible();
  await expect(gallery.getByRole('button').first()).toBeVisible();

  // No uncaught JS errors during load.
  expect(
    pageErrors,
    `uncaught page errors: ${pageErrors.join('; ')}`,
  ).toEqual([]);
});
