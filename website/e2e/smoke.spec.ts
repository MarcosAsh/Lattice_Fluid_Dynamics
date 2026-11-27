import { test, expect } from '@playwright/test';

// Loads the home page and checks the core UI mounts without an uncaught
// exception. A render-time crash (like the localStorage parse error that
// shipped once) throws before these elements appear, so this would catch it.
test('home page renders the control panel and demo gallery without crashing', async ({
  page,
}) => {
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
  expect(pageErrors, `uncaught page errors: ${pageErrors.join('; ')}`).toEqual(
    [],
  );
});

// Hash params and saved runs differ from the static HTML
test('shared params and saved runs hydrate without errors', async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  await page.addInitScript(() => {
    localStorage.setItem(
      'lattice_results',
      JSON.stringify([
        {
          videoUrl: '',
          cdValue: 0.4321,
          clValue: 0.1,
          cdSeries: [0.5, 0.4321],
          clSeries: [0.1, 0.1],
          model: 'ahmed25',
          windSpeed: 2,
          timestamp: 1,
        },
      ]),
    );
  });

  await page.goto('/#ws=3.5&m=ahmed35');

  await expect(page.getByText('3.5', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox').first()).toHaveValue('ahmed35');
  await expect(page.getByText('0.4321').first()).toBeVisible();
  expect(pageErrors, `uncaught page errors: ${pageErrors.join('; ')}`).toEqual(
    [],
  );
});

test('demo mode keeps the controls usable', async ({ page }) => {
  await page.goto('/');

  await expect(
    page.getByText('No GPU backend is connected, so rendering is off.', {
      exact: false,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Start Render' }),
  ).toBeDisabled();

  const wind = page.getByRole('slider').first();
  await expect(wind).toBeEnabled();
  await wind.fill('2.5');
  await expect(page).toHaveURL(/ws=2\.5/);

  await page.getByTestId('demo-gallery').getByRole('button').first().click();
  await expect(page.getByText('0.7772').first()).toBeVisible();
  await expect(page.getByTestId('demo-gallery')).toBeVisible();
});

test('viewing an older run shows its results', async ({ page }) => {
  await page.goto('/');
  const gallery = page.getByTestId('demo-gallery');
  await gallery.getByRole('button', { name: /^Car/ }).click();
  await gallery.getByRole('button', { name: /^Ahmed 25/ }).click();

  const dragCard = page.getByText('Drag Coefficient').locator('..');
  await expect(dragCard).toContainText('2.3002');

  await page.getByRole('button', { name: 'View' }).click();
  await expect(dragCard).toContainText('0.7772');
});

test('docs page is server rendered and tracks the section in view', async ({
  page,
  request,
}) => {
  const html = await (await request.get('/docs')).text();
  expect(html).toContain('Lattice Documentation');
  expect(html).toContain('cmake -B build -S simulation');

  await page.goto('/docs');
  const sidebar = page.locator('aside').first();
  await page.locator('#cli').scrollIntoViewIfNeeded();
  await expect(
    sidebar.getByRole('link', { name: 'CLI Reference' }),
  ).toHaveClass(/border-ctp-mauve/);
});

for (const [path, title] of [
  ['/', 'Lattice'],
  ['/docs', 'Documentation | Lattice'],
  ['/comparison', 'ML vs LBM | Lattice'],
  ['/optimize', 'Shape Optimization | Lattice'],
] as const) {
  test(`${path} has its title and loads cleanly`, async ({ page }) => {
    const problems: string[] = [];
    page.on('pageerror', (err) => problems.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') problems.push(msg.text());
    });

    await page.goto(path);
    await expect(page).toHaveTitle(title);
    await expect(page.locator('meta[name="theme-color"]')).toHaveAttribute(
      'content',
      '#1e1e2e',
    );
    await page.waitForLoadState('networkidle');
    expect(problems, problems.join('; ')).toEqual([]);
  });
}
