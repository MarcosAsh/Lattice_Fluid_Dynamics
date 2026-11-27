import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ControlPanel from '../src/components/ControlPanel';
import { DEFAULT_PARAMS } from '../src/lib/hashParams';

function setup(props: Partial<Parameters<typeof ControlPanel>[0]> = {}) {
  const setParams = vi.fn();
  render(
    <ControlPanel
      params={DEFAULT_PARAMS}
      setParams={setParams}
      onRender={() => {}}
      onSweep={() => {}}
      disabled={false}
      objFile={null}
      onObjFileChange={() => {}}
      {...props}
    />,
  );
  return { setParams };
}

describe('ControlPanel', () => {
  it('labels every control', () => {
    setup();
    for (const name of [
      'Model',
      'Wind Speed',
      'Duration',
      'Reynolds Number',
      'Visualization Mode',
      'Collision Mode',
      'Min sweep wind speed',
      'Max sweep wind speed',
      'Step sweep wind speed',
    ]) {
      expect(screen.getByLabelText(name)).toBeInTheDocument();
    }
  });

  it('describes the Reynolds preset in words', () => {
    setup({ params: { ...DEFAULT_PARAMS, reynolds: 500 } });
    expect(screen.getByLabelText('Reynolds Number')).toHaveAttribute(
      'aria-valuetext',
      '500, transitional',
    );
  });

  it('keeps inputs usable when rendering is unavailable', () => {
    const { setParams } = setup({
      canRender: false,
      renderHint: 'No GPU backend is connected.',
    });
    expect(screen.getByRole('button', { name: 'Start Render' })).toBeDisabled();
    expect(
      screen.getByText('No GPU backend is connected.'),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Wind Speed'), {
      target: { value: '2.5' },
    });
    expect(setParams).toHaveBeenCalledWith({
      ...DEFAULT_PARAMS,
      windSpeed: 2.5,
    });
  });

  it('exposes the mobile toggle state', () => {
    setup();
    const toggle = screen.getByRole('button', { name: 'Controls' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
  });
});
