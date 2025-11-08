# Salience Simulation Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Exploratory simulations for playing with salience-weighted gravity heuristics. This project is **not** an empirical discovery—it's a sandbox for experimenting with ideas and sharing reproducible code.

## ⚠️ Research Status & Disclaimers

- The models in this repository are speculative heuristics. They are designed to generate synthetic rotation curves and diagnostics for discussion, not to claim that a new physics theory has been proven.
- Outputs should be treated like **thought experiments**. Anyone using the code should validate results independently before drawing scientific conclusions.
- Historical artefacts are archived in `/archive/legacy/` so the main tree reflects the cleaner, neutral toolkit.

## ELI5: What does this do?

Imagine you are trying to guess how fast stars orbit in a galaxy. The usual recipe uses the galaxy's visible mass. This lab pretends that the *"attention"* a galaxy grabs might also matter—things like how smooth it looks, how coherent the spiral arms are, or how dense the gas is. We turn those impressions into numbers (we call the mix **salience**) and feed them into a calculator that makes synthetic rotation curves. It's basically a playground for "what if gravity cared about aesthetics?"

## What’s Included

- **Core library (`src/csg_v4/`)** – Salience scoring, rotation curve synthesizer, and configuration objects.
- **Synthetic data tools** – Helpers for generating toy galaxies and scanning calibration constants.
- **Diagnostics & reporting** – Scripts that produce markdown/JSON artefacts summarizing runs.
- **Experimental notebooks & docs** – Narrative explanations stored in `docs/` and cross-referenced in `AGENTS.md`.

Everything is wired to be reproducible: each script writes its inputs, config, and outputs under `artifacts/` so the paper trail is explicit.

## Getting Started

### Requirements

- Python 3.10+
- NumPy, Matplotlib, Pandas, Tabulate, tqdm (installable via `pip install -e .[dev]`)

### Quick Installation

```bash
cd /path/to/salience-simulation-lab
pip install -e .            # library only
pip install -e .[dev]       # with lint/test extras
```

### Run the synthetic pipeline

```bash
python -m csg_v4 --output-dir artifacts/latest_run
```

This loads three toy galaxies, sweeps the salience calibration, and writes:

1. synthetic rotation curves
2. diagnostic plots/tables
3. JSON + markdown summaries for auditing

### Poking the API

```python
from csg_v4 import CSGV4Model, CSGConfig
from csg_v4.synthetic_data import load_synthetic_galaxies

model = CSGV4Model()
galaxies = load_synthetic_galaxies()
galaxy = galaxies["NGC6503_like"]

salience = model.compute_salience(galaxy)
prediction = model.predict_velocity(galaxy, kappa_c=0.5)
print(prediction.v_pred[:5])
```

Use `scan_kappa` to sweep the calibration parameter and inspect residuals.

## Project Structure

```
salience-simulation-lab/
├── src/csg_v4/          # Core salience sandbox package
├── scripts/             # Command-line utilities and diagnostics
├── docs/                # Concept notes, diagrams, experiment logs
├── artifacts/           # Auto-generated outputs (kept under version control)
├── AGENTS.md            # Rolling experiment tracker
├── README.md            # This document
└── archive/legacy/      # Historical zips and prior versions (read-only)
```

## Salience Heuristic (Non-rigorous TL;DR)

- **Salience S'** multiplies together continuity, retention, coherence, and mass scores and then adjusts them with boosts/penalties for morphology and disorder.
- **Quality factor Q** is a smoothed version of salience that blends local and global impressions.
- **Acceleration estimate** interpolates between familiar Newtonian dynamics and a MOND-style square-root regime using Q and a calibration constant κ.

Mathematically these are just programmable knobs. Nothing in this repository asserts that nature behaves this way—it only checks what synthetic curves look like if we pretend it does.

## Documentation Map

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md) – Expanded explanation of the scoring pipeline.
- [`docs/examples/`](docs/examples/) – Walkthrough notebooks.
- [`AGENTS.md`](AGENTS.md) – High-level log of experiments, what changed, and which artefacts to inspect.

## Configuration Cheatsheet

Key parameters live in `CSGConfig`:

| Group            | Example knobs                     | Notes |
|------------------|------------------------------------|-------|
| Physical scales  | `G`, `a0`, `V0`                    | Defaults chosen for stable numerics |
| Salience weights | `alpha_n`, `gamma_W`, `beta_phi`   | Control boosts/penalties |
| Quality factor   | `aura_mix`, `min_s_prime`, `min_q_final` | Prevent degenerate values |

Override them by instantiating a custom config and passing it into `CSGV4Model`.

## Experimental Scripts

Selected helpers in `scripts/`:

- `continuity_mass_sim.py` – Stress test continuity weights.
- `time_dilation_train.py` – Toy experiments with continuity strain.
- `pack_gr_metrics.py` – Aggregate diagnostics across synthetic seeds.
- `gr_window_sweep.py` – Sensitivity sweep over Jacobian windows.

Each script writes artefacts into `artifacts/`; refer to `AGENTS.md` for run history and parameters.

## Contributing & Safety Notes

- Contributions are welcome if they keep the speculative nature clear and preserve reproducibility.
- Generated artefacts are tracked for transparency—please avoid editing them manually.
- If you extend the models, document the change in `AGENTS.md` and consider flagging anything that looks empirically promising for independent review.

## License & Citation

Specify your preferred license in `LICENSE`. If you use the code, cite it as:

```
Salience Simulation Lab (2025)
https://github.com/<your-handle>/salience-simulation-lab
```

## Questions?

- Open an issue
- Reach out via the contact listed in `AGENTS.md`
- Or simply explore the artefacts and see what the sandbox produces

---

**Reminder:** This project is a curiosity-driven sandbox. Treat every output as a starting point for discussion, not as evidence of a discovered phenomenon.
