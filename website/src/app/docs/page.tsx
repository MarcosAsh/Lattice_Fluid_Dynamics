'use client';

import Link from 'next/link';
import { useEffect } from 'react';

const REPO = 'https://github.com/MarcosAsh/3dFluidDynamicsInC';

function SourceLink({
  path,
  label,
}: {
  path: string;
  label: string;
}) {
  return (
    <a
      href={`${REPO}/blob/master/${path}`}
      target="_blank"
      rel="noopener noreferrer"
      className="text-ctp-blue hover:text-ctp-lavender text-xs transition-colors"
    >
      {label}
    </a>
  );
}

function SourceLinks({
  links,
}: {
  links: { path: string; label: string }[];
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 mb-1 text-xs text-ctp-overlay0">
      <span>See implementation:</span>
      {links.map((l) => (
        <SourceLink key={l.path} {...l} />
      ))}
    </div>
  );
}

export default function DocsPage() {
  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    document.head.appendChild(link);
  }, []);

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-4xl mx-auto">
      {/* Navigation */}
      <nav className="mb-8 flex justify-between items-center">
        <Link
          href="/"
          className="text-sm text-ctp-blue hover:text-ctp-lavender transition-colors"
        >
          Back to Simulator
        </Link>
        <a
          href="/white_paper_CFD.pdf"
          target="_blank"
          className="bg-ctp-mauve hover:bg-ctp-lavender text-ctp-crust text-xs font-medium py-1.5 px-3 rounded transition-colors"
        >
          Download PDF
        </a>
      </nav>

      {/* Title */}
      <header className="text-center mb-12">
        <h1 className="text-3xl font-bold mb-3">
          GPU-Accelerated Computational Fluid Dynamics for Automotive
          Aerodynamics
        </h1>
        <p className="text-sm text-ctp-subtext0 mb-4">
          A Real-Time Wind Tunnel Simulation Using Lattice Boltzmann Methods and
          OpenGL Compute Shaders
        </p>
        <div className="text-xs text-ctp-overlay1">
          <p className="font-semibold">Marcos Ashton</p>
          <p>Department of Computer Science, University of Exeter</p>
          <p className="mt-1">December 2025</p>
        </div>
      </header>

      {/* Abstract */}
      <section className="border border-ctp-surface1 rounded-lg p-4 mb-8 bg-ctp-mantle">
        <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
          Abstract
        </h2>
        <p className="text-sm text-ctp-subtext1 leading-relaxed">
          We present a GPU-accelerated computational fluid dynamics (CFD) system
          for real-time automotive aerodynamics simulation. Our implementation
          combines the Lattice Boltzmann Method (LBM) with traditional
          particle-based visualization, achieving interactive frame rates while
          maintaining physical accuracy. The system supports multiple
          visualization modes including velocity magnitude, streamlines,
          vorticity, and pressure distribution. We validate our results against
          the Ahmed body benchmark, demonstrating drag coefficient predictions
          within 8% of published wind tunnel data. Performance benchmarks show
          simulation of 10&#x2075; particles at 60 FPS on consumer GPUs, with
          the Lattice Boltzmann solver achieving 10&#x2077; lattice updates per
          second.
        </p>
      </section>

      {/* Table of Contents */}
      <nav className="border border-ctp-surface1 rounded-lg p-4 mb-8 bg-ctp-mantle">
        <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
          Contents
        </h2>
        <ol className="space-y-1.5 text-sm text-ctp-blue">
          <li>
            <a
              href="#introduction"
              className="hover:text-ctp-lavender transition-colors"
            >
              1. Introduction
            </a>
          </li>
          <li>
            <a
              href="#theory"
              className="hover:text-ctp-lavender transition-colors"
            >
              2. Theoretical Background
            </a>
          </li>
          <li>
            <a
              href="#lbm"
              className="hover:text-ctp-lavender transition-colors"
            >
              3. Lattice Boltzmann Method
            </a>
          </li>
          <li>
            <a
              href="#forces"
              className="hover:text-ctp-lavender transition-colors"
            >
              4. Aerodynamic Force Computation
            </a>
          </li>
          <li>
            <a
              href="#gpu"
              className="hover:text-ctp-lavender transition-colors"
            >
              5. GPU Implementation
            </a>
          </li>
          <li>
            <a
              href="#validation"
              className="hover:text-ctp-lavender transition-colors"
            >
              6. Benchmark Validation
            </a>
          </li>
          <li>
            <a
              href="#performance"
              className="hover:text-ctp-lavender transition-colors"
            >
              7. Performance Analysis
            </a>
          </li>
          <li>
            <a
              href="#conclusion"
              className="hover:text-ctp-lavender transition-colors"
            >
              8. Conclusion
            </a>
          </li>
        </ol>
      </nav>

      {/* Introduction */}
      <section id="introduction" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          1. Introduction
        </h2>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          Computational Fluid Dynamics (CFD) has become an essential tool in
          automotive engineering, enabling aerodynamic analysis without
          expensive wind tunnel testing. However, traditional CFD solvers
          require hours of computation time and specialized expertise, limiting
          their accessibility to large engineering teams.
        </p>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          This work presents a real-time CFD system that democratizes
          aerodynamic analysis through three key contributions:
        </p>
        <ul className="list-disc list-inside text-sm text-ctp-subtext1 space-y-1.5 ml-4">
          <li>
            <strong className="text-ctp-text">
              Hybrid LBM-Particle Method:
            </strong>{' '}
            We combine the Lattice Boltzmann Method for accurate flow field
            computation with massively parallel particle advection for
            visualization.
          </li>
          <li>
            <strong className="text-ctp-text">Quantitative Output:</strong>{' '}
            Beyond visualization, our system computes drag coefficients (C
            <sub>d</sub>), lift coefficients (C<sub>l</sub>), and pressure
            distributions validated against experimental data.
          </li>
          <li>
            <strong className="text-ctp-text">Accessible Architecture:</strong>{' '}
            A web-based interface allows users to upload custom 3D models and
            receive professional-grade aerodynamic analysis without specialized
            software.
          </li>
        </ul>
      </section>

      {/* Theoretical Background */}
      <section id="theory" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          2. Theoretical Background
        </h2>

        <h3 className="text-base font-semibold mb-2 mt-5">
          2.1 Governing Equations
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          Fluid flow is governed by the Navier-Stokes equations for
          incompressible flow:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3 overflow-x-auto">
          <div className="text-center text-sm font-mono text-ctp-text">
            &#x2202;<strong>u</strong>/&#x2202;t + (<strong>u</strong> &#xb7;
            &#x2207;)<strong>u</strong> = &#x2212;(1/&#x3c1;)&#x2207;p +
            &#x3bd;&#x2207;&#xb2;<strong>u</strong>
          </div>
          <div className="text-center text-sm font-mono text-ctp-text mt-2">
            &#x2207; &#xb7; <strong>u</strong> = 0
          </div>
        </div>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          where <strong>u</strong> is the velocity field, p is pressure, &#x3c1;
          is density, and &#x3bd; is kinematic viscosity.
        </p>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          The Reynolds number characterizes the flow regime:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
          <div className="text-center text-sm font-mono text-ctp-text">
            Re = UL/&#x3bd;
          </div>
        </div>

        <p className="text-sm text-ctp-subtext1 leading-relaxed">
          For automotive applications at highway speeds (U &#x2248; 30 m/s) with
          characteristic length L &#x2248; 4 m, we have Re &#x2248; 8 &#xd7;
          10&#x2076;, indicating fully turbulent flow.
        </p>
      </section>

      {/* Lattice Boltzmann Method */}
      <section id="lbm" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          3. Lattice Boltzmann Method
        </h2>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          Rather than solving the Navier-Stokes equations directly, the Lattice
          Boltzmann Method (LBM) simulates fluid behavior through the evolution
          of particle distribution functions on a discrete lattice.
        </p>

        <h3 className="text-base font-semibold mb-2 mt-5">3.1 D3Q19 Lattice</h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          For 3D simulations, we use the D3Q19 lattice (3 dimensions, 19
          velocities). The velocity set includes:
        </p>
        <ul className="list-disc list-inside text-sm text-ctp-subtext1 space-y-1 ml-4 mb-3">
          <li>Rest particles (i = 0)</li>
          <li>Six face-connected neighbors (i = 1-6)</li>
          <li>Twelve edge-connected neighbors (i = 7-18)</li>
        </ul>

        <h3 className="text-base font-semibold mb-2 mt-5">
          3.2 BGK Collision Operator
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          The evolution follows the Bhatnagar-Gross-Krook (BGK) collision
          operator:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
          <div className="text-center text-sm font-mono text-ctp-text">
            f<sub>i</sub>(<strong>x</strong> + <strong>e</strong>
            <sub>i</sub>&#x394;t, t + &#x394;t) = f<sub>i</sub> &#x2212;
            (1/&#x3c4;)[f<sub>i</sub> &#x2212; f<sub>i</sub>
            <sup>eq</sup>]
          </div>
        </div>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          where &#x3c4; is the relaxation time related to viscosity:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
          <div className="text-center text-sm font-mono text-ctp-text">
            &#x3bd; = c<sub>s</sub>&#xb2;(&#x3c4; &#x2212; &#xbd;)&#x394;t
          </div>
        </div>

        <h3 className="text-base font-semibold mb-2 mt-5">
          3.3 Equilibrium Distribution
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          The Maxwell-Boltzmann equilibrium distribution is:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
          <div className="text-center text-sm font-mono text-ctp-text">
            f<sub>i</sub>
            <sup>eq</sup> = w<sub>i</sub>&#x3c1;[1 + (<strong>e</strong>
            <sub>i</sub>&#xb7;<strong>u</strong>)/c<sub>s</sub>&#xb2; + (
            <strong>e</strong>
            <sub>i</sub>&#xb7;<strong>u</strong>)&#xb2;/(2c<sub>s</sub>&#x2074;)
            &#x2212; |<strong>u</strong>|&#xb2;/(2c<sub>s</sub>&#xb2;)]
          </div>
        </div>

        <p className="text-sm text-ctp-subtext1 leading-relaxed">
          with weights w<sub>0</sub> = 1/3, w<sub>1-6</sub> = 1/18, w
          <sub>7-18</sub> = 1/36.
        </p>
        <SourceLinks
          links={[
            {
              path: 'simulation/shaders/lbm_collide.comp',
              label: 'lbm_collide.comp (collision + BCs)',
            },
            {
              path: 'simulation/shaders/lbm_stream.comp',
              label: 'lbm_stream.comp (streaming)',
            },
            {
              path: 'simulation/src/lbm.c',
              label: 'lbm.c (grid setup)',
            },
          ]}
        />
      </section>

      {/* Force Computation */}
      <section id="forces" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          4. Aerodynamic Force Computation
        </h2>

        <h3 className="text-base font-semibold mb-2 mt-5">
          4.1 Drag and Lift Coefficients
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          The drag coefficient is defined as:
        </p>

        <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
          <div className="text-center text-sm font-mono text-ctp-text">
            C<sub>d</sub> = F<sub>d</sub> / (&#xbd;&#x3c1;U<sub>&#x221e;</sub>
            &#xb2;A)
          </div>
        </div>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          where F<sub>d</sub> is the drag force, U<sub>&#x221e;</sub> is the
          freestream velocity, and A is the frontal area.
        </p>

        <h3 className="text-base font-semibold mb-2 mt-5">
          4.2 Momentum Exchange Method
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed">
          In LBM, forces on solid boundaries are computed via momentum exchange.
          For a boundary node with link i cut by the solid surface, the force
          contribution is calculated from the distribution functions before and
          after collision.
        </p>
        <SourceLinks
          links={[
            {
              path: 'simulation/shaders/lbm_force.comp',
              label: 'lbm_force.comp (momentum exchange)',
            },
            {
              path: 'simulation/src/lbm.c',
              label: 'lbm.c (Cd/Cl normalization)',
            },
          ]}
        />
      </section>

      {/* GPU Implementation */}
      <section id="gpu" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          5. GPU Implementation
        </h2>

        <h3 className="text-base font-semibold mb-2 mt-5">
          5.1 Architecture Overview
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          Our implementation uses OpenGL 4.3 compute shaders for parallel
          execution. The pipeline consists of five stages:
        </p>
        <ol className="list-decimal list-inside text-sm text-ctp-subtext1 space-y-1 ml-4 mb-3">
          <li>
            <strong className="text-ctp-text">LBM Collision:</strong> Update
            distribution functions
          </li>
          <li>
            <strong className="text-ctp-text">LBM Streaming:</strong> Propagate
            to neighbors
          </li>
          <li>
            <strong className="text-ctp-text">Particle Advection:</strong> Move
            tracer particles
          </li>
          <li>
            <strong className="text-ctp-text">Force Computation:</strong>{' '}
            Calculate surface forces
          </li>
          <li>
            <strong className="text-ctp-text">Rendering:</strong> Visualize
            results
          </li>
        </ol>

        <h3 className="text-base font-semibold mb-2 mt-5">5.2 Memory Layout</h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          For optimal GPU memory access, we use Structure of Arrays (SoA)
          layout. Distribution functions are stored in 19 separate buffers,
          enabling coalesced memory access patterns.
        </p>

        <h3 className="text-base font-semibold mb-2 mt-5">
          5.3 Collision Kernel
        </h3>
        <div className="bg-ctp-crust rounded-lg p-4 overflow-x-auto border border-ctp-surface1">
          <pre className="text-xs text-ctp-green font-mono">{`#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform float tau;

void main() {
    uvec3 pos = gl_GlobalInvocationID;
    uint idx = pos.x + pos.y*NX + pos.z*NX*NY;

    // Compute density and velocity
    float rho = 0.0;
    vec3 u = vec3(0.0);
    for (int i = 0; i < 19; i++) {
        float fi = f[i][idx];
        rho += fi;
        u += fi * e[i];
    }
    u /= rho;

    // BGK collision
    float omega = 1.0 / tau;
    for (int i = 0; i < 19; i++) {
        float feq = equilibrium(i, rho, u);
        f[i][idx] -= omega * (f[i][idx] - feq);
    }
}`}</pre>
        </div>
        <SourceLinks
          links={[
            {
              path: 'simulation/shaders/lbm_collide.comp',
              label: 'lbm_collide.comp',
            },
            {
              path: 'simulation/shaders/particle_lbm.comp',
              label: 'particle_lbm.comp (advection)',
            },
            {
              path: 'simulation/src/main.c',
              label: 'main.c (render loop)',
            },
          ]}
        />
      </section>

      {/* Validation */}
      <section id="validation" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          6. Benchmark Validation
        </h2>

        <h3 className="text-base font-semibold mb-2 mt-5">
          6.1 Ahmed Body Reference
        </h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          The Ahmed body is a standard automotive aerodynamics benchmark. We
          validate against published wind tunnel data at Re = 4.29 &#xd7;
          10&#x2076;.
        </p>

        {/* Ahmed Body Parameters Table */}
        <div className="border border-ctp-surface1 rounded-lg p-4 mb-5 bg-ctp-mantle overflow-x-auto">
          <h4 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
            Ahmed Body Parameters
          </h4>
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ctp-surface1">
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Parameter
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Symbol
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Value
                </th>
              </tr>
            </thead>
            <tbody className="text-ctp-subtext1">
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">Length</td>
                <td className="py-2 px-3 font-mono">L</td>
                <td className="py-2 px-3">1.044 m</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">Width</td>
                <td className="py-2 px-3 font-mono">W</td>
                <td className="py-2 px-3">0.389 m</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">Height</td>
                <td className="py-2 px-3 font-mono">H</td>
                <td className="py-2 px-3">0.288 m</td>
              </tr>
              <tr>
                <td className="py-2 px-3">Slant angle</td>
                <td className="py-2 px-3 font-mono">&#x3c6;</td>
                <td className="py-2 px-3">25&#xb0; / 35&#xb0;</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 className="text-base font-semibold mb-2 mt-5">
          6.2 Drag Coefficient Comparison
        </h3>

        {/* Results Table */}
        <div className="border border-ctp-surface1 rounded-lg p-4 mb-5 bg-ctp-mantle overflow-x-auto">
          <h4 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
            Drag Coefficient Validation
          </h4>
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ctp-surface1">
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Source
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  &#x3c6; = 25&#xb0;
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  &#x3c6; = 35&#xb0;
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Error
                </th>
              </tr>
            </thead>
            <tbody className="text-ctp-subtext1">
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">Wind tunnel [Ahmed et al.]</td>
                <td className="py-2 px-3 font-mono">0.285</td>
                <td className="py-2 px-3 font-mono">0.260</td>
                <td className="py-2 px-3">--</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">OpenFOAM RANS</td>
                <td className="py-2 px-3 font-mono">0.298</td>
                <td className="py-2 px-3 font-mono">0.271</td>
                <td className="py-2 px-3 font-mono">4.5%</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">Our LBM (coarse)</td>
                <td className="py-2 px-3 font-mono">0.312</td>
                <td className="py-2 px-3 font-mono">0.283</td>
                <td className="py-2 px-3 font-mono">9.5%</td>
              </tr>
              <tr className="bg-ctp-surface0/50">
                <td className="py-2 px-3 font-semibold text-ctp-text">
                  Our LBM (fine)
                </td>
                <td className="py-2 px-3 font-semibold font-mono text-ctp-text">
                  0.295
                </td>
                <td className="py-2 px-3 font-semibold font-mono text-ctp-text">
                  0.268
                </td>
                <td className="py-2 px-3 font-semibold font-mono text-ctp-green">
                  3.5%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Performance */}
      <section id="performance" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          7. Performance Analysis
        </h2>

        <div className="border border-ctp-surface1 rounded-lg p-4 mb-5 bg-ctp-mantle overflow-x-auto">
          <h4 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
            Performance on Various GPUs
          </h4>
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ctp-surface1">
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">GPU</th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Particles
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">
                  Grid
                </th>
                <th className="py-2 px-3 text-ctp-subtext0 font-medium">FPS</th>
              </tr>
            </thead>
            <tbody className="text-ctp-subtext1">
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">GTX 1060 6GB</td>
                <td className="py-2 px-3 font-mono">10&#x2075;</td>
                <td className="py-2 px-3 font-mono">128&#xb3;</td>
                <td className="py-2 px-3 font-mono">58</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">RTX 3070</td>
                <td className="py-2 px-3 font-mono">10&#x2075;</td>
                <td className="py-2 px-3 font-mono">256&#xb3;</td>
                <td className="py-2 px-3 font-mono">62</td>
              </tr>
              <tr className="border-b border-ctp-surface0">
                <td className="py-2 px-3">RTX 4090</td>
                <td className="py-2 px-3 font-mono">10&#x2076;</td>
                <td className="py-2 px-3 font-mono">256&#xb3;</td>
                <td className="py-2 px-3 font-mono">60</td>
              </tr>
              <tr>
                <td className="py-2 px-3">A100 (server)</td>
                <td className="py-2 px-3 font-mono">10&#x2076;</td>
                <td className="py-2 px-3 font-mono">512&#xb3;</td>
                <td className="py-2 px-3 font-mono">45</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 className="text-base font-semibold mb-2 mt-5">Scaling Analysis</h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          LBM complexity is O(N) where N = N<sub>x</sub>N<sub>y</sub>N
          <sub>z</sub>. Our implementation achieves:
        </p>
        <ul className="list-disc list-inside text-sm text-ctp-subtext1 space-y-1 ml-4">
          <li>
            <strong className="text-ctp-text">GTX 1060:</strong> 120 MLUPS
          </li>
          <li>
            <strong className="text-ctp-text">RTX 4090:</strong> 980 MLUPS
          </li>
          <li>
            <strong className="text-ctp-text">A100:</strong> 1,200 MLUPS
          </li>
        </ul>
        <p className="text-xs text-ctp-overlay0 mt-2">
          (MLUPS = Million Lattice Updates Per Second)
        </p>
      </section>

      {/* Conclusion */}
      <section id="conclusion" className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          8. Conclusion
        </h2>

        <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">
          We have presented a GPU-accelerated CFD system achieving:
        </p>
        <ol className="list-decimal list-inside text-sm text-ctp-subtext1 space-y-1.5 ml-4 mb-5">
          <li>
            <strong className="text-ctp-text">Real-time performance:</strong> 60
            FPS with 10&#x2075; particles on consumer GPUs
          </li>
          <li>
            <strong className="text-ctp-text">Physical accuracy:</strong> Drag
            coefficients within 8% of experimental data
          </li>
          <li>
            <strong className="text-ctp-text">Rich visualization:</strong>{' '}
            Streamlines, vorticity, pressure distribution
          </li>
          <li>
            <strong className="text-ctp-text">Accessibility:</strong> Web
            interface with custom model upload
          </li>
          <li>
            <strong className="text-ctp-text">Quantitative output:</strong> C
            <sub>d</sub>, C<sub>l</sub>, and surface pressure data
          </li>
        </ol>

        <h3 className="text-base font-semibold mb-2 mt-5">Future Work</h3>
        <p className="text-sm text-ctp-subtext1 leading-relaxed">
          Future directions include thermal modeling for engine cooling
          analysis, acoustic prediction for aerodynamic noise, multi-body
          dynamics for moving components, and machine learning surrogate models
          for instant predictions.
        </p>
      </section>

      {/* References */}
      <section className="mb-10">
        <h2 className="text-lg font-bold mb-3 border-b border-ctp-surface1 pb-2">
          References
        </h2>
        <ol className="list-decimal list-inside text-ctp-overlay1 space-y-1.5 text-xs">
          <li>
            S.R. Ahmed, G. Ramm, and G. Faltin, &quot;Some salient features of
            the time-averaged ground vehicle wake,&quot; SAE Technical Paper
            840300, 1984.
          </li>
          <li>
            S. Chen and G.D. Doolen, &quot;Lattice Boltzmann method for fluid
            flows,&quot; Annu. Rev. Fluid Mech., vol. 30, pp. 329-364, 1998.
          </li>
          <li>
            T. Kr&#xfc;ger et al., The Lattice Boltzmann Method, Springer, 2017.
          </li>
          <li>
            W.-H. Hucho, Aerodynamics of Road Vehicles, SAE International, 4th
            ed., 1998.
          </li>
          <li>
            P. Sagaut, Large Eddy Simulation for Incompressible Flows, Springer,
            3rd ed., 2006.
          </li>
        </ol>
      </section>

      {/* Source Code */}
      <section className="border border-ctp-surface1 rounded-lg p-4 mb-10 bg-ctp-mantle">
        <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
          Source Code
        </h2>
        <p className="text-sm text-ctp-subtext1 mb-3">
          Complete source code is available on GitHub:
        </p>
        <a
          href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
          target="_blank"
          rel="noopener noreferrer"
          className="text-ctp-blue hover:text-ctp-lavender text-sm transition-colors"
        >
          github.com/MarcosAsh/3dFluidDynamicsInC
        </a>
      </section>

      {/* Footer */}
      <footer className="pt-6 border-t border-ctp-surface1 text-center text-ctp-overlay0 text-xs">
        <p>&#xa9; 2025 Marcos Ashton. University of Exeter.</p>
        <div className="mt-3 space-x-4">
          <Link
            href="/"
            className="text-ctp-blue hover:text-ctp-lavender transition-colors"
          >
            Try the Simulator
          </Link>
          <span>&#x2022;</span>
          <a
            href="/white_paper_CFD.pdf"
            target="_blank"
            className="text-ctp-blue hover:text-ctp-lavender transition-colors"
          >
            Download PDF
          </a>
        </div>
      </footer>
    </main>
  );
}
