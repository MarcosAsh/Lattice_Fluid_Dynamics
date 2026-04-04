'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { SECTION_GROUPS, ALL_SECTIONS } from '../../lib/docs-sections';
import { DocsSidebar } from '../../components/docs/DocsSidebar';
import { CodeBlock } from '../../components/docs/CodeBlock';
import { InlineCode } from '../../components/docs/InlineCode';
import { Callout } from '../../components/docs/Callout';
import { DocTable } from '../../components/docs/DocTable';
import { MathBlock } from '../../components/docs/MathBlock';
import { SrcLink } from '../../components/docs/SrcLink';
import { GridCalculator } from '../../components/docs/GridCalculator';

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
    <section id={id} className="mb-16 scroll-mt-20">
      <h2 className="text-xl font-bold mb-5 pb-3 border-b border-ctp-surface1">
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
      { rootMargin: '-20% 0px -70% 0px' },
    );
    for (const s of ALL_SECTIONS) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen flex">
      <DocsSidebar groups={SECTION_GROUPS} activeSection={active} />

      <main className="flex-1 max-w-3xl mx-auto px-6 lg:px-10 py-8">
        {/* Breadcrumb */}
        <nav className="mb-8 text-xs text-ctp-overlay1 flex items-center gap-2">
          <Link
            href="/"
            className="hover:text-ctp-text transition-colors"
          >
            Home
          </Link>
          <span>/</span>
          <span className="text-ctp-text">Documentation</span>
        </nav>

        <header className="mb-12">
          <h1 className="text-3xl font-bold mb-3">Lattice Documentation</h1>
          <P>
            GPU-accelerated wind tunnel simulation using Lattice Boltzmann
            Methods. Runs on consumer GPUs, outputs drag and lift coefficients,
            and includes an ML surrogate for instant predictions.
          </P>
        </header>

        {/* ------------------------------------------------------------ */}
        {/* Quick Start                                                   */}
        {/* ------------------------------------------------------------ */}
        <Section id="quickstart" title="Quick Start">
          <Callout type="tip">
            Requires OpenGL 4.3+ and a GPU with compute shader support. Any
            discrete NVIDIA or AMD card from the last 8 years will work.
          </Callout>

          <h3 className="text-base font-semibold mb-3 mt-6">
            Build from source
          </h3>
          <CodeBlock language="bash" title="terminal">{`# Dependencies (Ubuntu/Debian)
sudo apt install build-essential cmake pkg-config \\
  libsdl2-dev libsdl2-ttf-dev mesa-common-dev \\
  libgl1-mesa-dev libegl1-mesa-dev

# Build
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run interactively
cd simulation
../build/3d_fluid_simulation_car --wind=1.0`}</CodeBlock>

          <h3 className="text-base font-semibold mb-3 mt-8">Headless render</h3>
          <CodeBlock language="bash" title="terminal">{`mkdir -p frames
../build/3d_fluid_simulation_car \\
  --wind=2.0 --duration=10 \\
  --model=assets/3d-files/ahmed_25deg_m.obj \\
  --output=frames --grid=256x128x128`}</CodeBlock>

          <h3 className="text-base font-semibold mb-3 mt-8">Web frontend</h3>
          <CodeBlock language="bash" title="terminal">{`cd website
npm ci && npm run dev
# Open http://localhost:3000`}</CodeBlock>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* How It Works                                                  */}
        {/* ------------------------------------------------------------ */}
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
            and Bouzidi interpolated bounce-back on solid walls. The Bouzidi scheme
            stores the fractional distance from each fluid cell to the actual mesh
            surface along every lattice link, then applies quadratic or linear
            interpolation during streaming. This gives second-order wall accuracy
            without refining the grid.
          </P>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Architecture                                                  */}
        {/* ------------------------------------------------------------ */}
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
      particle.comp      # Particle advection + collision
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

        {/* ------------------------------------------------------------ */}
        {/* CLI Reference                                                 */}
        {/* ------------------------------------------------------------ */}
        <Section id="cli" title="CLI Reference">
          <DocTable
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
              ['-c, --collision=MODE', '2', 'Collision: 0=off, 1=AABB, 2=mesh, 3=voxel'],
            ]}
          />
          <Callout type="note">
            Wind speed controls Reynolds number when <InlineCode>--reynolds</InlineCode> is
            not set: Re = 200 * windSpeed, capped by the grid{"'"}s minimum stable
            tau (0.52). Larger grids support higher Re. Voxel collision (mode 3)
            requires LBM and will fall back to AABB if LBM is disabled.
          </Callout>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* API Reference                                                 */}
        {/* ------------------------------------------------------------ */}
        <Section id="api" title="API Reference">
          <P>
            The web frontend talks to a Modal GPU worker via a REST endpoint.
          </P>
          <h3 className="text-base font-semibold mb-3">POST /api/render</h3>
          <CodeBlock language="json" title="request">{`{
  "wind_speed": 1.0,
  "duration": 10,
  "model": "ahmed25",
  "viz_mode": 1,
  "collision_mode": 2,
  "reynolds": 0
}`}</CodeBlock>
          <CodeBlock language="json" title="response">{`{
  "status": "complete",
  "video_url": "https://...",
  "cd_value": 0.295,
  "cl_value": -0.012,
  "cd_series": [0.31, 0.30, 0.295],
  "effective_re": 480,
  "grid_size": "256x128x128"
}`}</CodeBlock>
          <P>
            The worker runs on an NVIDIA T4 GPU via Modal. Cold starts take
            30-60 seconds; warm starts respond in ~15 seconds for a 10-second
            simulation.
          </P>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Visualization Modes                                           */}
        {/* ------------------------------------------------------------ */}
        <Section id="visualization" title="Visualization Modes">
          <DocTable
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
            Press <InlineCode>V</InlineCode> to cycle through modes in interactive mode.
            Streamline mode (7) stores a 32-frame position trail per particle and
            renders as GL_LINE_STRIP with velocity-based coloring and alpha fade.
          </P>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Physics Details                                               */}
        {/* ------------------------------------------------------------ */}
        <Section id="physics" title="Physics Details">
          <h3 className="text-base font-semibold mb-3">Governing Equations</h3>
          <P>
            Fluid flow is governed by the Navier-Stokes equations for
            incompressible flow:
          </P>
          <MathBlock>
            &#x2202;<strong>u</strong>/&#x2202;t + (<strong>u</strong> &#xb7;
            &#x2207;)<strong>u</strong> = &#x2212;(1/&#x3c1;)&#x2207;p +
            &#x3bd;&#x2207;&#xb2;<strong>u</strong>
          </MathBlock>
          <MathBlock>
            &#x2207; &#xb7; <strong>u</strong> = 0
          </MathBlock>
          <P>
            where <strong>u</strong> is the velocity field, p is pressure,
            &#x3c1; is density, and &#x3bd; is kinematic viscosity. The Reynolds
            number Re = UL/&#x3bd; characterizes the flow regime. For automotive
            aerodynamics at highway speeds, Re reaches 10&#x2076; or higher,
            indicating fully turbulent flow.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">
            Lattice Boltzmann Method
          </h3>
          <P>
            Rather than solving Navier-Stokes directly, the LBM evolves particle
            distribution functions on a D3Q19 lattice (3 dimensions, 19
            velocities): 1 rest, 6 face-connected, and 12 edge-connected
            neighbors. The BGK collision operator relaxes distributions toward
            equilibrium:
          </P>
          <MathBlock>
            f<sub>i</sub>(<strong>x</strong> + <strong>e</strong>
            <sub>i</sub>&#x394;t, t + &#x394;t) = f<sub>i</sub> &#x2212;
            (1/&#x3c4;)[f<sub>i</sub> &#x2212; f<sub>i</sub>
            <sup>eq</sup>]
          </MathBlock>
          <Callout type="note">
            The relaxation time &#x3c4; controls viscosity: &#x3bd; =
            c<sub>s</sub>&#xb2;(&#x3c4; &#x2212; &#xbd;)&#x394;t. Lower
            &#x3c4; means higher Re but less numerical stability. The minimum
            stable value is &#x3c4; = 0.52.
          </Callout>

          <h3 className="text-base font-semibold mb-3 mt-8">
            Equilibrium Distribution
          </h3>
          <P>
            The Maxwell-Boltzmann equilibrium distribution is:
          </P>
          <MathBlock>
            f<sub>i</sub>
            <sup>eq</sup> = w<sub>i</sub>&#x3c1;[1 + (<strong>e</strong>
            <sub>i</sub>&#xb7;<strong>u</strong>)/c<sub>s</sub>&#xb2; + (
            <strong>e</strong>
            <sub>i</sub>&#xb7;<strong>u</strong>)&#xb2;/(2c<sub>s</sub>
            &#x2074;) &#x2212; |<strong>u</strong>|&#xb2;/(2c<sub>s</sub>
            &#xb2;)]
          </MathBlock>
          <P>
            with weights w<sub>0</sub> = 1/3, w<sub>1-6</sub> = 1/18, w
            <sub>7-18</sub> = 1/36 and speed of sound c<sub>s</sub>&#xb2; = 1/3.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">
            Collision Kernel
          </h3>
          <CodeBlock language="glsl" title="lbm_collide.comp (simplified)">{`#version 430 core
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

          <h3 className="text-base font-semibold mb-3 mt-8">
            Smagorinsky SGS Model
          </h3>
          <P>
            For turbulent flows, the Smagorinsky subgrid-scale model computes
            a local eddy viscosity from the non-equilibrium stress tensor:
          </P>
          <MathBlock>
            &#x3c4;<sub>eff</sub> = &#xbd;(&#x3c4; +
            &#x221a;(&#x3c4;&#xb2; + 18C<sub>s</sub>&#xb2;|&#x3a0;|/&#x3c1;))
          </MathBlock>
          <P>
            where C<sub>s</sub> is the Smagorinsky constant (default 0.1,
            tunable via <InlineCode>--smagorinsky</InlineCode>) and |&#x3a0;| is the
            Frobenius norm of the non-equilibrium stress. This increases
            &#x3c4; locally in turbulent regions, preventing instability without
            damping the entire domain.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">
            Drag Computation
          </h3>
          <P>
            Surface forces use the Mei-Luo-Shyy momentum exchange method,
            which pairs with the Bouzidi boundary to account for the actual
            wall position rather than assuming it sits on the grid midpoint:
          </P>
          <MathBlock>
            <strong>F</strong> = &#x2211;<sub>i</sub> <strong>e</strong>
            <sub>i</sub> (f<sub>i</sub>*(<strong>x</strong><sub>f</sub>) +
            f<sub>i&#x0304;</sub>(<strong>x</strong><sub>f</sub>))
          </MathBlock>
          <P>
            where f<sub>i</sub>* is the post-collision distribution heading
            toward the wall and f<sub>i&#x0304;</sub> is the Bouzidi-reflected
            distribution coming back. The drag and lift coefficients are then:
          </P>
          <MathBlock>
            C<sub>d</sub> = F<sub>x</sub> / (&#xbd;&#x3c1;U&#xb2;A)
            &nbsp;&nbsp;&nbsp;&nbsp;
            C<sub>l</sub> = F<sub>y</sub> / (&#xbd;&#x3c1;U&#xb2;A)
          </MathBlock>
          <div className="flex gap-3 mt-2 text-xs">
            <SrcLink path="simulation/shaders/lbm_collide.comp" label="lbm_collide.comp" />
            <SrcLink path="simulation/shaders/lbm_stream.comp" label="lbm_stream.comp" />
            <SrcLink path="simulation/shaders/lbm_force.comp" label="lbm_force.comp" />
          </div>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* ML Surrogate Model                                            */}
        {/* ------------------------------------------------------------ */}
        <Section id="ml" title="ML Surrogate Model">
          <P>
            A small MLP predicts Cd and Cl in under a millisecond, giving
            instant estimates before the LBM has time to converge. The entire
            training framework is written from scratch in C++ with reverse-mode
            autodiff, so there are no Python ML dependencies at inference time.
            The same model runs in the C simulation binary, in the browser via
            TypeScript, and in the evaluation pipeline.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">Architecture</h3>
          <P>
            The network takes three z-score normalized inputs (wind speed,
            Reynolds number, model ID) and outputs two values (Cd, Cl). SwiGLU
            activations replace standard ReLU for smoother gradient flow.
          </P>
          <CodeBlock>{`Linear(3, 256) -> SwiGLU(256, 512) -> Linear(256, 128) -> SwiGLU(128, 256) -> Linear(128, 2)

525,400 parameters total
  fc1:  3 x 256  + bias     =   1,024
  act1: SwiGLU(256, 512)    = 393,216  (gate + up + down projections)
  fc2:  256 x 128 + bias    =  32,896
  act2: SwiGLU(128, 256)    =  98,304
  fc3:  128 x 2 + bias      =     258`}</CodeBlock>

          <h3 className="text-base font-semibold mb-3 mt-8">Training</h3>
          <P>
            Training data comes from the LBM simulation itself. A parameter
            sweep over wind speeds, Reynolds numbers, and model geometries
            generates (input, Cd, Cl) pairs. The data generator
            (<InlineCode>data_gen.py</InlineCode>) submits render jobs to Modal, validates
            convergence, and writes a binary dataset.
          </P>
          <CodeBlock>{`Optimizer:    AdamW (lr=1e-3, weight_decay=1e-4, clip_norm=1.0)
Loss:         MSE on [Cd, Cl]
Batch size:   64
Epochs:       500 (early stopping, patience=50)
Train/val:    80/20 split
Data format:  binary -- 16-byte header + N records of 5 float32s
              [wind_speed, reynolds, model_id, cd, cl]`}</CodeBlock>

          <h3 className="text-base font-semibold mb-3 mt-8">Weight Format</h3>
          <P>
            Weights are stored in a custom binary format (LTWS) shared across
            all three inference targets. The normalizer file stores 3 means and
            3 standard deviations as raw float32s.
          </P>
          <CodeBlock>{`model.bin:      LTWS header (magic + version + count) + 12 parameter tensors
model_norm.bin: 6 x float32 (mean[3], std[3])`}</CodeBlock>

          <h3 className="text-base font-semibold mb-3 mt-8">Inference</h3>
          <P>
            The C simulation loads the weights at startup and prints an instant
            Cd/Cl estimate before the LBM loop begins. The browser loads the
            same files from <InlineCode>/models/</InlineCode> and runs a pure TypeScript
            forward pass with no dependencies. Both paths take under 1ms.
          </P>
          <CodeBlock language="bash" title="terminal">{`# Generate training data
cd ml && python data_gen.py --config sweep_config.yaml --endpoint $MODAL_URL

# Train the model
cmake -B build && cmake --build build && ./build/train dataset/training_data.bin

# Evaluate
python evaluate.py --weights model.bin --norm model_norm.bin --data dataset/training_data.bin`}</CodeBlock>
          <div className="flex gap-3 mt-2 text-xs flex-wrap">
            <SrcLink path="ml/train.cpp" label="train.cpp" />
            <SrcLink path="ml/framework/src/autodiff.cpp" label="autodiff.cpp" />
            <SrcLink path="ml/framework/src/layers/ad_swiglu.cpp" label="ad_swiglu.cpp" />
            <SrcLink path="simulation/src/ml_predict.c" label="ml_predict.c" />
            <SrcLink path="website/src/lib/surrogate.ts" label="surrogate.ts" />
            <SrcLink path="ml/data_gen.py" label="data_gen.py" />
          </div>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Validation                                                    */}
        {/* ------------------------------------------------------------ */}
        <Section id="validation" title="Validation">
          <P>
            The solver uses a sphere drag test to validate the force computation
            pipeline (momentum exchange, Bouzidi boundary, Cd formula). At Re=100
            with 16 cells across the sphere diameter, the measured Cd falls
            within the expected range from Clift, Grace &amp; Weber (1978).
          </P>

          <h3 className="text-base font-semibold mb-3 mt-6">
            Sphere reference test
          </h3>
          <DocTable
            headers={['Re', 'Grid', 'Cd (sim)', 'Cd (ref)', 'Status']}
            rows={[
              ['100', '128\u00d764\u00d764', '1.1 \u2013 1.3', '1.09', 'Pass (30% tol)'],
            ]}
          />

          <h3 className="text-base font-semibold mb-3 mt-6">
            Why Cd differs from wind tunnel data
          </h3>
          <P>
            Published Ahmed body data (Ahmed 1984, Lienhart 2003) is measured
            at Re &gt; 500,000 where the flow is fully turbulent. The LBM
            solver on typical grids (128-256 cells) operates at Re = 50-200
            where the flow is laminar. At laminar Re the boundary layer is
            thicker, separation occurs earlier, and Cd is 3-10x higher
            than at turbulent Re. This is physically correct behavior, not
            a bug.
          </P>
          <P>
            The Smagorinsky SGS turbulence model adds local eddy viscosity that
            partially compensates, but matching high-Re experimental data requires
            grids with hundreds of cells across the body -- beyond what consumer
            GPUs can handle in real time.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-6">
            Grid size and Reynolds number
          </h3>
          <P>
            The achievable Reynolds number is limited by the grid resolution.
            The relaxation parameter &tau; must stay above 0.52 for numerical
            stability. This relationship determines the maximum Re for each
            grid:
          </P>
          <MathBlock>
            Re<sub>max</sub> = U &#xb7; L<sub>body</sub> / &#x3bd;<sub>min</sub>
            &nbsp;&nbsp;&nbsp;&nbsp;
            &#x3bd;<sub>min</sub> = (&#x3c4;<sub>min</sub> &#x2212; 0.5) / 3
          </MathBlock>
          <DocTable
            headers={['Grid', 'Body cells', 'Re (max)', 'Cd range', 'GPU VRAM']}
            rows={[
              ['128\u00d764\u00d764', '~38', '~115', '2 \u2013 5', '86 MB'],
              ['128\u00d796\u00d796', '~38', '~115', '2 \u2013 5', '240 MB'],
              ['256\u00d7128\u00d7128', '~77', '~230', '1 \u2013 3', '960 MB'],
              ['512\u00d7256\u00d7256', '~153', '~460', '0.5 \u2013 2', '7.6 GB'],
            ]}
          />
          <Callout type="note">
            The Cd values above are for the Ahmed body at 25{'\u00b0'} slant.
            Published wind tunnel Cd {'\u2248'} 0.29 is at Re {'\u2248'} 768,000
            (Lienhart 2003). Reaching that Re would need {'\u2248'}2,000 cells
            across the body.
          </Callout>

          <h3 className="text-base font-semibold mb-3 mt-6">
            Try it yourself
          </h3>
          <P>
            Use the calculator below to see what Reynolds number and memory
            footprint your grid configuration produces:
          </P>
          <GridCalculator />
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Performance                                                   */}
        {/* ------------------------------------------------------------ */}
        <Section id="performance" title="Performance">
          <P>
            Performance depends on grid size and GPU. The LBM kernel is
            memory-bandwidth-bound: each cell reads and writes 19 float32
            distributions per step. Typical throughput on modern GPUs:
          </P>
          <DocTable
            headers={['GPU', 'Grid', 'Steps/sec', 'MLUPS']}
            rows={[
              ['A10G (Modal)', '128\u00d796\u00d796', '~400', '~470'],
              ['A100 40GB (Modal)', '256\u00d7128\u00d7128', '~100', '~420'],
              ['RTX 3070', '128\u00d764\u00d764', '~800', '~420'],
              ['RTX 4090', '256\u00d7128\u00d7128', '~200', '~840'],
            ]}
          />
          <P>
            MLUPS = Million Lattice Updates Per Second. Memory usage per cell is
            approximately 200 bytes (19 distributions {'\u00d7'} 2 buffers +
            velocity + solid + Bouzidi q). A 256{'\u00d7'}128{'\u00d7'}128 grid
            needs about 960 MB of GPU buffer space.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-6">
            Convergence time
          </h3>
          <P>
            The drag coefficient needs 3-5 flow-throughs to converge. A
            flow-through is the time for a fluid element to traverse the domain:
            N_x / U_lattice steps. For a 256-cell domain at U = 0.05, one
            flow-through takes 5,120 steps. The simulation reports an exponential
            moving average (EMA) of Cd that stabilizes after approximately 50
            samples.
          </P>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* Contributing                                                  */}
        {/* ------------------------------------------------------------ */}
        <Section id="contributing" title="Contributing">
          <P>
            Lattice is open source under the MIT license. Contributions are
            welcome -- see{' '}
            <a
              href="https://github.com/MarcosAsh/3dFluidDynamicsInC/blob/master/CONTRIBUTING.md"
              target="_blank"
              rel="noopener noreferrer"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              CONTRIBUTING.md
            </a>{' '}
            for the full style guide, testing, and PR process.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">Code style</h3>
          <P>
            The simulation is C11, formatted with clang-format (LLVM style,
            80 char limit). The website is TypeScript/React with Tailwind.
            Tests run in CI via GitHub Actions.
          </P>

          <h3 className="text-base font-semibold mb-3 mt-8">Commit sign-off</h3>
          <P>
            Every commit must carry a sign-off line certifying the{' '}
            <a
              href="https://developercertificate.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-ctp-blue hover:text-ctp-lavender transition-colors"
            >
              Developer Certificate of Origin
            </a>
            . Use <InlineCode>git commit -s</InlineCode> to add it automatically:
          </P>
          <CodeBlock language="bash">{`git commit -s -m "lbm: fix bounce-back at concave corners"

# Produces:
# lbm: fix bounce-back at concave corners
#
# Signed-off-by: Your Name <your@email.com>`}</CodeBlock>
          <Callout type="warning">
            Commits without a valid sign-off will be rejected by CI.
            Make sure your Git name and email match your GitHub account.
          </Callout>

          <h3 className="text-base font-semibold mb-3 mt-8">Running tests</h3>
          <CodeBlock language="bash" title="terminal">{`# Run tests
cd simulation && xvfb-run ../build/test_lbm  # LBM unit tests
xvfb-run bash test/test_integration.sh       # Integration test
cd website && npm test                        # Frontend tests
cd ml/framework && ./build/test_ml            # ML framework tests`}</CodeBlock>
        </Section>

        {/* ------------------------------------------------------------ */}
        {/* References                                                    */}
        {/* ------------------------------------------------------------ */}
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
              href="https://github.com/MarcosAsh/3dFluidDynamicsInC"
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
