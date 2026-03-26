export const REPO = 'https://github.com/MarcosAsh/3dFluidDynamicsInC';

export interface DocSection {
  id: string;
  label: string;
}

export interface DocGroup {
  label: string;
  sections: DocSection[];
}

export const SECTION_GROUPS: DocGroup[] = [
  {
    label: 'Getting Started',
    sections: [
      { id: 'quickstart', label: 'Quick Start' },
      { id: 'how-it-works', label: 'How It Works' },
      { id: 'architecture', label: 'Architecture' },
    ],
  },
  {
    label: 'Reference',
    sections: [
      { id: 'cli', label: 'CLI Reference' },
      { id: 'api', label: 'API Reference' },
      { id: 'visualization', label: 'Visualization Modes' },
    ],
  },
  {
    label: 'Theory & Validation',
    sections: [
      { id: 'physics', label: 'Physics Details' },
      { id: 'ml', label: 'ML Surrogate Model' },
      { id: 'validation', label: 'Validation' },
      { id: 'performance', label: 'Performance' },
    ],
  },
  {
    label: 'Community',
    sections: [{ id: 'contributing', label: 'Contributing' }],
  },
];

export const ALL_SECTIONS: DocSection[] = SECTION_GROUPS.flatMap(
  (g) => g.sections,
);
