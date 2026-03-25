import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ResultsPanel, {
  SimulationResult,
} from '../src/components/ResultsPanel';
import StatusDisplay from '../src/components/StatusDisplay';

const mockResult: SimulationResult = {
  videoUrl: 'https://example.com/video.mp4',
  cdValue: 0.287,
  clValue: 0.041,
  cdSeries: [0.31, 0.29, 0.28],
  clSeries: [0.04, 0.04, 0.04],
  model: 'ahmed25',
  windSpeed: 1.5,
  charLength: 0.044,
  timestamp: Date.now(),
};

describe('ResultsPanel', () => {
  it('renders nothing when history is empty', () => {
    const { container } = render(
      <ResultsPanel
        current={null}
        history={[]}
        onSelect={() => {}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('displays Cd and Cl values', () => {
    render(
      <ResultsPanel
        current={mockResult}
        history={[mockResult]}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByText('0.2870')).toBeInTheDocument();
    expect(screen.getByText('0.0410')).toBeInTheDocument();
  });

  it('shows -- for null Cd/Cl', () => {
    const nullResult: SimulationResult = {
      ...mockResult,
      cdValue: null,
      clValue: null,
    };
    render(
      <ResultsPanel
        current={nullResult}
        history={[nullResult]}
        onSelect={() => {}}
      />,
    );
    const dashes = screen.getAllByText('--');
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });

  it('shows model label', () => {
    render(
      <ResultsPanel
        current={mockResult}
        history={[mockResult]}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByText('Ahmed 25')).toBeInTheDocument();
  });
});

describe('StatusDisplay', () => {
  it('shows Ready when idle', () => {
    render(
      <StatusDisplay
        status="idle"
        error={null}
        duration={10}
        renderStartTime={null}
      />,
    );
    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('shows Complete when done', () => {
    render(
      <StatusDisplay
        status="complete"
        error={null}
        duration={10}
        renderStartTime={null}
      />,
    );
    expect(
      screen.getByText('Complete!'),
    ).toBeInTheDocument();
  });

  it('shows Error when failed', () => {
    render(
      <StatusDisplay
        status="error"
        error="GPU crashed"
        duration={10}
        renderStartTime={null}
      />,
    );
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(
      screen.getByText('GPU crashed'),
    ).toBeInTheDocument();
  });
});
