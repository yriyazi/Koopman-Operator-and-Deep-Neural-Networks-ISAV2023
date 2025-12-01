# Duffing Solution Library

Welcome to the high-energy playground where numerical integration, chaos theory, and data storytelling collide. This library bundles everything you need to **generate**, **analyze**, and **visualize** rich dynamical behaviors of the Duffing oscillator, supporting the broader Koopman-operator research found in the root of this repository.

---

## Highlights

- **Precision Integrators**: Fourth-order Runge-Kutta solvers with progress reporting and metadata-aware saving.
- **Dataset Factory**: Produce long-horizon trajectories across forcing amplitudes, damping coefficients, initial conditions, and sampling frequencies. Outputs are stored as `.npy` files with descriptive filenames for instant traceability.
- **Exploratory Notebooks**: Three ready-to-run Jupyter notebooks walk through single-trajectory analysis, bulk dataset generation, and advanced Poincare-map studies.
- **Visualization Arsenal**: Time-series plots, 2D/3D phase portraits, animated phase sweeps, and stroboscopic return maps ready for publications or presentations.
- **Reusable Utilities**: Helper modules for plotting, configuration, and pipeline automation so that new experiments slot right in.

---

## Notebooks

| Notebook | Focus | Key Capabilities |
| --- | --- | --- |
| `Duffing_Oscillator_RungeKutta_Analysis.ipynb` | Exploratory analysis | Single-run simulations, parameter sweeps, 3D phase space visualization, quick comparisons between forcing regimes |
| `Duffing_Dataset_Generator.ipynb` | Data manufacturing | Systematic trajectory generation, multi-frequency sampling, metadata-aware storage in `datasets/`, ready-made inputs for Koopman or ML pipelines |
| `Duffing_Poincare_Map_Generator.ipynb` | Chaos diagnostics | Dense Poincare maps, animation exports, batch runs across hundreds of initial conditions |

Each notebook begins with a detailed markdown primer describing purpose, mathematical background, and the outputs it creates.

---

## Key Modules

- `dataloaders/Runge_Kutta.py`: Core integrators (`runge_kutta_step`, `runge_kutta_solve`, `solve_and_plot`) that drive all simulations.
- `datasets/duffing_euqation.py`: Canonical Duffing ODE definition returning displacement/velocity derivatives for NumPy-powered integration.
- `utils/plot.py`: Consistent plotting helpers used by notebooks and scripts to keep visualization styles uniform.
- `poncare_scater.py`, `poncare_scater_all.py`: Animation scripts for high-resolution phase portraits and frame-by-frame return-map exploration.
- `results/`: Pre-rendered figures and animations grouped by analysis flavor (general solution, 3D phase plane, Poincare section).

---

## Dataset Layout

Generated files inside `datasets/` follow this pattern:

```
gamma=0.37 t_span=(0, 50000) initial_conditions=[1.5, -1.5] step_frequency=010.npy
```

Each filename captures:

- `gamma`: forcing amplitude
- `t_span`: integration horizon
- `initial_conditions`: starting displacement and velocity
- `step_frequency`: samples per unit time, zero-padded for stable sorting

This convention makes it trivial to select datasets programmatically based on scenario, cadence, or initial state.

---

## Quick Start

1. Install dependencies (NumPy, Matplotlib, tqdm, optional ffmpeg for GIF/video export).
2. Open any notebook in this directory and run top-to-bottom—parameters are set for chaotic regimes out of the box.
3. Browse the `datasets/` directory for ready-made trajectories or regenerate them with your preferred settings.
4. Use `utils/plot.py` and the animation scripts to turn raw arrays into publication-grade figures.

For command-line usage, import `dataloaders.runge_kutta_solve` inside your own scripts to integrate new scenarios programmatically.

---

## Where to Go Next

- Feed generated datasets into the Koopman and neural architectures under `../Loss` and `../Deeplearning`.
- Extend the solvers with stochastic forces, parameter scheduling, or alternative nonlinear terms.
- Customize the visualization scripts to highlight bifurcation diagrams or Lyapunov exponent estimates.

Whether you are teaching chaos theory, validating Koopman operators, or preparing ML-ready synthetic datasets, this library keeps Duffing dynamics at your fingertips.