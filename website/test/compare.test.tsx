import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import RunComparison from '../src/components/RunComparison';
import ResultsPanel from '../src/components/ResultsPanel';
import {
  compareMetrics,
  compareParams,
  comparisonCsv,
} from '../src/lib/compare';
import { DEFAULT_PARAMS } from '../src/lib/hashParams';
import type { SimulationResult } from '../src/lib/types';

function run(over: Partial<SimulationResult>): SimulationResult {
  return {
    videoUrl: 'https://example.com/a.mp4',
    cdValue: 0.5,
    clValue: 0.1,
    cdSeries: [0.6, 0.5],
    clSeries: [0.1, 0.1],
    charLength: null,
    sampleInterval: null,
    tStar: null,
    flowThroughs: null,
    cfl: 0.05,
    cdPressureSeries: [],
    cdFrictionSeries: [],
    model: 'ahmed25',
    windSpeed: 1,
    timestamp: 1,
    ...over,
  };
}

const a = run({ params: { ...DEFAULT_PARAMS, model: 'ahmed25' } });
const b = run({
  model: 'ahmed35',
  cdValue: 0.6,
  clValue: null,
  cdSeries: [0.7, 0.65, 0.6],
  videoUrl: '',
  timestamp: 2,
  params: { ...DEFAULT_PARAMS, model: 'ahmed35', reynolds: 500 },
});

describe('compareMetrics', () => {
  it('computes differences against run A', () => {
    const cd = compareMetrics(a, b).find((m) => m.key === 'cd');
    expect(cd?.diff).toBeCloseTo(0.1);
    expect(cd?.pct).toBeCloseTo(20);
  });

  it('leaves the difference empty when a side is missing', () => {
    const cl = compareMetrics(a, b).find((m) => m.key === 'cl');
    expect(cl).toMatchObject({ a: 0.1, b: null, diff: null, pct: null });
  });
});

describe('compareParams', () => {
  it('flags the parameters that differ', () => {
    const differs = compareParams(a, b)
      .filter((p) => p.differs)
      .map((p) => p.key);
    expect(differs).toEqual(['model', 'reynolds']);
  });

  it('says when an older run has no stored params', () => {
    const old = run({ timestamp: 3 });
    const re = compareParams(old, b).find((p) => p.key === 'reynolds');
    expect(re).toMatchObject({ a: 'Not recorded', b: '500', differs: true });
  });
});

describe('comparisonCsv', () => {
  it('aligns both series by step', () => {
    const lines = comparisonCsv(a, b).split('\n');
    expect(lines).toContain('step,cd_a,cl_a,cd_b,cl_b');
    expect(lines.at(-1)).toBe('2,,,0.600000,');
    expect(lines[0]).toBe('# Run A: Ahmed 25° at 1.0 m/s');
  });
});

describe('RunComparison', () => {
  it('shows both runs with their values and differences', () => {
    render(<RunComparison a={a} b={b} onClose={() => {}} />);
    const panel = screen.getByTestId('run-comparison');
    expect(within(panel).getByText('Ahmed 25° at 1.0 m/s')).toBeInTheDocument();
    expect(within(panel).getByText('Ahmed 35° at 1.0 m/s')).toBeInTheDocument();
    const cdRow = within(panel)
      .getByRole('rowheader', { name: 'Drag coefficient (Cd)' })
      .closest('tr');
    expect(cdRow).not.toBeNull();
    expect(
      within(cdRow as HTMLElement).getByText('+0.1000 (+20.0%)'),
    ).toBeInTheDocument();
    expect(
      within(panel).getByText('No video for this run'),
    ).toBeInTheDocument();
    expect(
      within(panel).getByRole('img', { name: /Cd over 3 measurement steps/ }),
    ).toBeInTheDocument();
  });

  it('calls onClose', () => {
    const onClose = vi.fn();
    render(<RunComparison a={a} b={b} onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Close comparison' }));
    expect(onClose).toHaveBeenCalled();
  });
});

describe('ResultsPanel compare selection', () => {
  it('lets the user tick at most two runs', () => {
    const c = run({ timestamp: 3, model: 'car' });
    const onToggle = vi.fn();
    render(
      <ResultsPanel
        current={c}
        history={[a, b, c]}
        onSelect={() => {}}
        compareIds={[1, 2]}
        onToggleCompare={onToggle}
      />,
    );
    const boxes = screen.getAllByRole('checkbox', { name: /^Compare / });
    expect(boxes).toHaveLength(3);
    const third = screen.getByRole('checkbox', {
      name: 'Compare Car at 1.0 m/s',
    });
    expect(third).toBeDisabled();
    const first = screen.getByRole('checkbox', {
      name: 'Compare Ahmed 25° at 1.0 m/s',
    });
    expect(first).toBeChecked();
    fireEvent.click(first);
    expect(onToggle).toHaveBeenCalledWith(1);
  });
});
