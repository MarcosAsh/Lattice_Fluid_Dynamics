'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

const REPO = 'https://github.com/MarcosAsh/3dFluidDynamicsInC';

const sections = [
  { id: 'quickstart', label: 'Quick Start' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'cli', label: 'CLI Reference' },
  { id: 'api', label: 'API Reference' },
  { id: 'visualization', label: 'Visualization Modes' },
  { id: 'ml', label: 'ML Surrogate Model' },
  { id: 'physics', label: 'Physics Details' },
  { id: 'validation', label: 'Validation' },
  { id: 'performance', label: 'Performance' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'contributing', label: 'Contributing' },
];

function Code({ children }: { children: string }) {
  return (
    <code className="bg-ctp-crust px-1.5 py-0.5 rounded text-ctp-green text-xs font-mono">
      {children}
    </code>
  );
}

function CodeBlock({ children, title }: { children: string; title?: string }) {
  return (
    <div className="bg-ctp-crust rounded-lg border border-ctp-surface1 mb-4 overflow-x-auto">
      {title && (
        <div className="text-[10px] text-ctp-overlay0 px-4 pt-2 uppercase tracking-wider">
          {title}
        </div>
      )}
      <pre className="text-xs text-ctp-text font-mono p-4 pt-2">{children}</pre>
    </div>
  );
}

function SrcLink({ path, label }: { path: string; label: string }) {
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

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="mb-12 scroll-mt-8">
      <h2 className="text-xl font-bold mb-4 pb-2 border-b border-ctp-surface1">
        {title}
      </h2>
      {children}
    </section>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-sm text-ctp-subtext1 leading-relaxed mb-3">{children}</p>
  );
}

function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | React.ReactNode)[][];
}) {
  return (
    <div className="border border-ctp-surface1 rounded-lg bg-ctp-mantle overflow-x-auto mb-4">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-ctp-surface1">
            {headers.map((h) => (
              <th
                key={h}
                className="py-2 px-3 text-ctp-subtext0 font-medium text-xs"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-ctp-subtext1 text-xs">
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-ctp-surface0 last:border-0">
              {row.map((cell, j) => (
                <td key={j} className="py-2 px-3 font-mono">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DocsPage() {
  const [active, setActive] = useState('quickstart');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActive(entry.target.id);
          }
        }
      },
      { rootMargin: '-20% 0px -70% 0px' }
    );
    for (const s of sections) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <nav className="hidden lg:block w-56 shrink-0 sticky top-0 h-screen overflow-y-auto p-6 border-r border-ctp-surface1">
        <Link
          href="/"
          className="text-sm font-bold text-ctp-text block mb-6"
        >
          Lattice
        </Link>
        <ul className="space-y-1">
          {sections.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className={`block text-xs py-1 px-2 rounded transition-colors ${
                  active === s.id
                    ? 'bg-ctp-surface0 text-ctp-text font-medium'
                    : 'text-ctp-overlay1 hover:text-ctp-text'
                }`}
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>
        <div className="mt-6 pt-4 border-t border-ctp-surface1 space-y-2">
          <a
            href={REPO}
            target="_blank"
            rel="noopener noreferrer"
            className="block text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors"
          >
            GitHub
          </a>
          <a
            href="/white_paper_CFD.pdf"
            target="_blank"
            className="block text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors"
          >
            White Paper (PDF)
          </a>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 max-w-3xl mx-auto p-6 lg:p-10">
        {/* Mobile nav */}
        <nav className="lg:hidden mb-6 flex justify-between items-center">
          <Link
            href="/"
            className="text-sm text-ctp-blue hover:text-ctp-lavender transition-colors"
          >
            Back to Simulator
          </Link>
        </nav>

        <header className="mb-10">
          <h1 className="text-2xl font-bold mb-2">Lattice Documentation</h1>
          <P>
            GPU-accelerated wind tunnel simulation using Lattice Boltzmann
            Methods. Runs on consumer GPUs, outputs drag and lift coefficients,
            and includes an ML surrogate for instant predictions.
          </P>
        </header>

        {/* Quick Start */}
        <Section id="quickstart" title="Quick Start">
          <h3 className="text-sm font-semibold mb-2">Build from source</h3>
          <CodeBlock title="terminal">{`# Dependencies (Ubuntu/Debian)
sudo apt install build-essential cmake pkg-config \\
  libsdl2-dev libsdl2-ttf-dev mesa-common-dev \\
  libgl1-mesa-dev libegl1-mesa-dev

# Build
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run interactively
cd simulation
../build/3d_fluid_simulation_car --wind=1.0`}</CodeBlock>

          <h3 className="text-sm font-semibold mb-2 mt-6">Headless render</h3>
          <CodeBlock title="terminal">{`mkdir -p frames
../build/3d_fluid_simulation_car \\
  --wind=2.0 --duration=10 \\
  --model=assets/3d-files/ahmed_25deg_m.obj \\
  --output=frames --grid=256x128x128`}</CodeBlock>

          <h3 className="text-sm font-semibold mb-2 mt-6">Web frontend</h3>
          <CodeBlock title="terminal">{`cd website
npm ci && npm run dev
# Open http://localhost:3000`}</CodeBlock>
        </Section>

        {/* How It Works */}
        <Section id="how-it-works" title="How It Works">
          <P>
            Each frame runs five GPU compute passes in sequence:
          </P>
          <ol className="list-decimal list-inside text-sm text-ctp-subtext1 space-y-2 mb-4 ml-2">
            <li>
              <strong className="text-ctp-text">Collision</strong> -- relax
              distribution functions toward equilibrium (BGK or regularized
              operator, with optional Smagorinsky SGS turbulence model)
            </li>
            <li>
              <strong className="text-ctp-text">Streaming</strong> -- propagate
              distributions to neighboring cells
            </li>
            <li>
              <strong className="text-ctp-text">Force computation</strong> --
              accumulate drag/lift via momentum exchange at solid boundaries
            </li>
            <li>
              <strong className="text-ctp-text">Particle advection</strong> --
              move tracer particles through the velocity field
            </li>
            <li>
              <strong className="text-ctp-text">Rendering</strong> -- draw
              particles as points or streamline trails
            </li>
          </ol>
          <P>
            The LBM solver uses a D3Q19 lattice (19 velocity directions in 3D).
            Solid geometry is voxelized from OBJ meshes using ray casting.
            Boundary conditions are Zou-He velocity inlet, Zou-He pressure outlet,
            and simple bounce-back on solid walls.
          </P>
        </Section>

        {/* CLI Reference */}
        <Section id="cli" title="CLI Reference">
          <Table
            headers={['Flag', 'Default', 'Description']}
            rows={[
              ['-w, --wind=SPEED', '1.0', 'Wind speed 0-5 (maps to Re)'],
              ['-d, --duration=SECS', '0', 'Headless render duration (0 = interactive)'],
              ['-o, --output=PATH', '', 'Output directory for PPM frames'],
              ['-m, --model=PATH', 'car-model.obj', 'Path to OBJ mesh file'],
              ['-a, --angle=DEG', '', 'Ahmed body slant angle (25 or 35)'],
              ['-g, --grid=XxYxZ', '128x64x64', 'LBM grid resolution'],
              ['-r, --reynolds=RE', '0', 'Target Reynolds number (0 = derive from wind)'],
              ['-s, --scale=S', '0.05', 'Model scale factor'],
              ['-S, --smagorinsky=CS', '0.1', 'Smagorinsky constant (0-0.5)'],
              ['-v, --viz=MODE', '1', 'Visualization mode (0-7)'],
              ['-c, --collision=MODE', '2', 'Collision: 0=off, 1=AABB, 2=mesh'],
            ]}
          />
          <P>
            Wind speed controls Reynolds number when <Code>--reynolds</Code> is
            not set: Re = 200 * windSpeed, capped by the grid{"'"}s minimum stable
            tau (0.52). Larger grids support higher Re.
          </P>
        </Section>

        {/* API Reference */}
        <Section id="api" title="API Reference">
          <P>
            The web frontend talks to a Modal GPU worker via a REST endpoint.
          </P>
          <h3 className="text-sm font-semibold mb-2">POST /api/render</h3>
          <CodeBlock title="request">{`{
  "wind_speed": 1.0,
  "duration": 10,
  "model": "ahmed25",      // "car" | "ahmed25" | "ahmed35"
  "viz_mode": 1,
  "collision_mode": 1,
  "reynolds": 0            // 0 = auto
}`}</CodeBlock>
          <CodeBlock title="response">{`{
  "status": "complete",
  "video_url": "https://...",
  "cd_value": 0.295,
  "cl_value": -0.012,
  "cd_series": [0.31, 0.30, 0.295, ...],
  "effective_re": 480,
  "grid_size": "256x128x128"
}`}</CodeBlock>
          <P>
            The worker runs on an NVIDIA T4 GPU via Modal. Cold starts take
            30-60 seconds; warm starts respond in ~15 seconds for a 10-second
            simulation.
          </P>
        </Section>

        {/* Visualization Modes */}
        <Section id="visualization" title="Visualization Modes">
          <Table
            headers={['Mode', 'Key', 'Description']}
            rows={[
              ['0', '3', 'Depth -- distance-based coloring'],
              ['1', '4', 'Velocity magnitude -- jet colormap (blue to red)'],
              ['2', '5', 'Velocity direction -- RGB = normalized XYZ'],
              ['3', '6', 'Particle lifetime -- viridis colormap with fade'],
              ['4', '7', 'Turbulence -- cool-warm diverging map'],
              ['5', '8', 'Flow progress -- rainbow by x-position'],
              ['6', '9', 'Vorticity -- purple (laminar) to orange (turbulent)'],
              [
                '7',
                'V',
                'Streamlines -- particle trails as line strips, 32-point history',
              ],
            ]}
          />
          <P>
            Press <Code>V</Code> to cycle through modes in interactive mode.
            Streamline mode (7) stores a 32-frame position trail per particle and
            renders as GL_LINE_STRIP with velocity-based coloring and alpha fade.
          </P>
        </Section>

        {/* ML Surrogate */}
        <Section id="ml" title="ML Surrogate Model">
          <P>
            A trained MLP predicts Cd and Cl in microseconds, giving instant
            estimates before the LBM converges. The model runs inside the
            simulation loop with zero external dependencies.
          </P>
          <h3 className="text-sm font-semibold mb-2">Architecture</h3>
          <CodeBlock>{`Input: [wind_speed, reynolds, model_id]  (3 features, z-score normalized)
  -> Linear(3, 256) -> SwiGLU(256, 512)
  -> Linear(256, 128) -> SwiGLU(128, 256)
  -> Linear(128, 2)
Output: [Cd, Cl]

Parameters: 525,698
Training: AdamW, MSE loss, 285 samples, early stopping
Validation MAE: Cd = 0.005, Cl = 0.002`}</CodeBlock>

          <h3 className="text-sm font-semibold mb-2 mt-4">Training</h3>
          <CodeBlock title="terminal">{`cd ml
cmake -B build && cmake --build build
python data_gen.py --config sweep_config.yaml --endpoint $MODAL_URL
./build/train dataset/training_data.bin`}</CodeBlock>

          <h3 className="text-sm font-semibold mb-2 mt-4">Inference</h3>
          <P>
            Place <Code>model.bin</Code> and <Code>model_norm.bin</Code> in the
            simulation working directory. The sim loads them automatically and
            prints an instant estimate at startup.
          </P>
          <div className="flex gap-3 mt-2 text-xs">
            <SrcLink path="ml/train.cpp" label="train.cpp" />
            <SrcLink path="simulation/src/ml_predict.c" label="ml_predict.c" />
            <SrcLink path="ml/data_gen.py" label="data_gen.py" />
          </div>
        </Section>

        {/* Physics Details */}
        <Section id="physics" title="Physics Details">
          <h3 className="text-sm font-semibold mb-2">Governing Equations</h3>
          <P>
            Fluid flow is governed by the Navier-Stokes equations for
            incompressible flow:
          </P>
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
          <P>
            where <strong>u</strong> is the velocity field, p is pressure,
            &#x3c1; is density, and &#x3bd; is kinematic viscosity. The Reynolds
            number Re = UL/&#x3bd; characterizes the flow regime. For automotive
            aerodynamics at highway speeds, Re reaches 10&#x2076; or higher,
            indicating fully turbulent flow.
          </P>

          <h3 className="text-sm font-semibold mb-2 mt-6">
            Lattice Boltzmann Method
          </h3>
          <P>
            Rather than solving Navier-Stokes directly, the LBM evolves particle
            distribution functions on a D3Q19 lattice (3 dimensions, 19
            velocities): 1 rest, 6 face-connected, and 12 edge-connected
            neighbors. The BGK collision operator relaxes distributions toward
            equilibrium:
          </P>
          <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
            <div className="text-center text-sm font-mono text-ctp-text">
              f<sub>i</sub>(<strong>x</strong> + <strong>e</strong>
              <sub>i</sub>&#x394;t, t + &#x394;t) = f<sub>i</sub> &#x2212;
              (1/&#x3c4;)[f<sub>i</sub> &#x2212; f<sub>i</sub>
              <sup>eq</sup>]
            </div>
          </div>
          <P>
            The relaxation time &#x3c4; controls viscosity: &#x3bd; =
            c<sub>s</sub>&#xb2;(&#x3c4; &#x2212; &#xbd;)&#x394;t. Lower
            &#x3c4; means higher Re but less numerical stability. The minimum
            stable value is &#x3c4; = 0.52.
          </P>

          <h3 className="text-sm font-semibold mb-2 mt-6">
            Equilibrium Distribution
          </h3>
          <P>
            The Maxwell-Boltzmann equilibrium distribution is:
          </P>
          <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
            <div className="text-center text-sm font-mono text-ctp-text">
              f<sub>i</sub>
              <sup>eq</sup> = w<sub>i</sub>&#x3c1;[1 + (<strong>e</strong>
              <sub>i</sub>&#xb7;<strong>u</strong>)/c<sub>s</sub>&#xb2; + (
              <strong>e</strong>
              <sub>i</sub>&#xb7;<strong>u</strong>)&#xb2;/(2c<sub>s</sub>
              &#x2074;) &#x2212; |<strong>u</strong>|&#xb2;/(2c<sub>s</sub>
              &#xb2;)]
            </div>
          </div>
          <P>
            with weights w<sub>0</sub> = 1/3, w<sub>1-6</sub> = 1/18, w
            <sub>7-18</sub> = 1/36 and speed of sound c<sub>s</sub>&#xb2; = 1/3.
          </P>

          <h3 className="text-sm font-semibold mb-2 mt-6">
            Collision Kernel
          </h3>
          <CodeBlock title="lbm_collide.comp (simplified)">{`#version 430 core
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
}`}</CodeBlock>

          <h3 className="text-sm font-semibold mb-2 mt-6">
            Smagorinsky SGS Model
          </h3>
          <P>
            For turbulent flows, the Smagorinsky subgrid-scale model computes
            a local eddy viscosity from the non-equilibrium stress tensor:
          </P>
          <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
            <div className="text-center text-sm font-mono text-ctp-text">
              &#x3c4;<sub>eff</sub> = &#xbd;(&#x3c4; +
              &#x221a;(&#x3c4;&#xb2; + 18C<sub>s</sub>&#xb2;|&#x3a0;|/&#x3c1;))
            </div>
          </div>
          <P>
            where C<sub>s</sub> is the Smagorinsky constant (default 0.1,
            tunable via <Code>--smagorinsky</Code>) and |&#x3a0;| is the
            Frobenius norm of the non-equilibrium stress. This increases
            &#x3c4; locally in turbulent regions, preventing instability without
            damping the entire domain.
          </P>

          <h3 className="text-sm font-semibold mb-2 mt-6">
            Drag Computation
          </h3>
          <P>
            Surface forces use Ladd{"'"}s (1994) momentum exchange method for
            stationary bounce-back boundaries:
          </P>
          <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
            <div className="text-center text-sm font-mono text-ctp-text">
              <strong>F</strong> = &#x2211;<sub>i</sub> 2<strong>e</strong>
              <sub>i</sub> f<sub>i</sub>*(<strong>x</strong><sub>f</sub>)
            </div>
          </div>
          <P>
            where f* is the post-collision distribution at the fluid cell
            adjacent to the solid surface. The drag and lift coefficients are
            then:
          </P>
          <div className="bg-ctp-surface0 rounded-lg p-4 mb-3">
            <div className="text-center text-sm font-mono text-ctp-text">
              C<sub>d</sub> = F<sub>x</sub> / (&#xbd;&#x3c1;U&#xb2;A)
              &nbsp;&nbsp;&nbsp;&nbsp;
              C<sub>l</sub> = F<sub>y</sub> / (&#xbd;&#x3c1;U&#xb2;A)
            </div>
          </div>
          <div className="flex gap-3 mt-2 text-xs">
            <SrcLink
              path="simulation/shaders/lbm_collide.comp"
              label="lbm_collide.comp"
            />
            <SrcLink
              path="simulation/shaders/lbm_stream.comp"
              label="lbm_stream.comp"
            />
            <SrcLink
              path="simulation/shaders/lbm_force.comp"
              label="lbm_force.comp"
            />
          </div>
        </Section>

        {/* Validation */}
        <Section id="validation" title="Validation">
          <P>
            Validated against the Ahmed body, a standard automotive aerodynamics
            benchmark (Ahmed et al., SAE 840300).
          </P>
          <Table
            headers={['Source', '\u03c6 = 25\u00b0', '\u03c6 = 35\u00b0', 'Error']}
            rows={[
              ['Wind tunnel (Ahmed et al.)', '0.285', '0.260', '--'],
              ['OpenFOAM RANS', '0.298', '0.271', '4.5%'],
              ['Lattice (coarse, 128\u00b3)', '0.312', '0.283', '9.5%'],
              ['Lattice (fine, 256\u00b3)', '0.295', '0.268', '3.5%'],
            ]}
          />
          <P>
            The fine-grid result matches within 3.5% of wind tunnel data. The
            coarse grid is less accurate but runs at interactive frame rates on
            consumer hardware.
          </P>
        </Section>

        {/* Performance */}
        <Section id="performance" title="Performance">
          <Table
            headers={['GPU', 'Grid', 'FPS', 'MLUPS']}
            rows={[
              ['GTX 1060 6GB', '128\u00b3', '58', '120'],
              ['RTX 3070', '256\u00b3', '62', '450'],
              ['RTX 4090', '256\u00b3', '60', '980'],
              ['NVIDIA T4 (Modal)', '256x128x128', 'headless', '~300'],
            ]}
          />
          <P>
            MLUPS = Million Lattice Updates Per Second. LBM complexity is O(N)
            where N is the total number of grid cells. Memory usage scales as
            ~150 bytes per cell (19 distributions x 2 buffers + velocity +
            solid mask).
          </P>
        </Section>

        {/* Architecture */}
        <Section id="architecture" title="Architecture">
          <CodeBlock>{`lattice/
  simulation/
    src/main.c           # Render loop, CLI, GL setup
    src/lbm.c            # LBM grid creation, force readback
    src/ml_predict.c     # ML inference forward pass (pure C)
    shaders/
      lbm_collide.comp   # BGK + Smagorinsky + BCs
      lbm_stream.comp    # Pull-based streaming
      lbm_force.comp     # Momentum exchange drag
      particle.comp      # Particle advection
      trail_update.comp  # Streamline trail history
    lib/                 # Headers
    assets/              # OBJ models, fonts
  website/
    src/app/page.tsx     # Main simulator UI
    src/app/api/         # Next.js API routes
    src/components/      # React components
  ml/
    train.cpp            # Training driver
    data_gen.py          # Modal sweep for training data
    framework/           # C++ autodiff, layers, optimizer`}</CodeBlock>
        </Section>

        {/* Contributing */}
        <Section id="contributing" title="Contributing">
          <P>
            See{' '}
            <a
              href={`${REPO}/blob/master/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              CONTRIBUTING.md
            </a>{' '}
            for style guide, testing, and PR process. The simulation is C11,
            formatted with clang-format (LLVM style, 80 char limit). The website
            is TypeScript/React with Tailwind. Tests run in CI via GitHub
            Actions.
          </P>
          <CodeBlock title="terminal">{`# Run tests
cd simulation && xvfb-run ../build/test_lbm  # LBM unit tests
xvfb-run bash test/test_integration.sh       # Integration test
cd website && npm test                        # Frontend tests
cd ml/framework && ./build/test_ml            # ML framework tests`}</CodeBlock>
        </Section>

        {/* References */}
        <section className="mb-10 text-xs text-ctp-overlay1">
          <h3 className="text-sm font-semibold text-ctp-subtext0 mb-2">
            References
          </h3>
          <ol className="list-decimal list-inside space-y-1">
            <li>
              Ahmed, Ramm, Faltin. &quot;Some salient features of the
              time-averaged ground vehicle wake.&quot; SAE 840300, 1984.
            </li>
            <li>
              Chen, Doolen. &quot;Lattice Boltzmann method for fluid flows.&quot;
              Annu. Rev. Fluid Mech., 1998.
            </li>
            <li>
              Kr&uuml;ger et al. The Lattice Boltzmann Method. Springer, 2017.
            </li>
            <li>
              Ladd. &quot;Numerical simulations of particulate suspensions via a
              discretized Boltzmann equation.&quot; J. Fluid Mech., 1994.
            </li>
          </ol>
        </section>

        {/* Footer */}
        <footer className="pt-6 border-t border-ctp-surface1 text-center text-ctp-overlay0 text-xs">
          <p>&copy; 2025-2026 Marcos Ashton. University of Exeter.</p>
          <div className="mt-3 space-x-4">
            <Link
              href="/"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              Simulator
            </Link>
            <span>&bull;</span>
            <a
              href={REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              GitHub
            </a>
            <span>&bull;</span>
            <a
              href="/white_paper_CFD.pdf"
              target="_blank"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              White Paper
            </a>
          </div>
        </footer>
      </main>
    </div>
  );
}
