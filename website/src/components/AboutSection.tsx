'use client';

export default function AboutSection() {
  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        About
      </h2>
      <p className="text-xs text-ctp-overlay1 leading-relaxed mb-3">
        GPU-accelerated wind tunnel simulation visualizing particle flow around
        3D models using OpenGL compute shaders and the Lattice Boltzmann Method.
        Computes physically accurate drag coefficients validated against
        experimental data.
      </p>
      <p className="text-xs text-ctp-subtext0 mb-3">
        Built by Marcos Ashton, CS at University of Exeter.
      </p>
      <a
        href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
        target="_blank"
        rel="noopener noreferrer"
        className="text-ctp-blue hover:text-ctp-lavender text-xs transition-colors"
      >
        Source Code
      </a>
    </div>
  );
}
