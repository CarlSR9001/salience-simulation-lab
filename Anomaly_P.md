# Anomaly Report – Experiment P (Extreme Continuity Stress Test)

## Overview
Experiment P explores continuity-taxed PID controllers under hostile forcing (Gaussian noise, Lorenz chaotic coupling) to test whether programmable inertia tracks control energy. Across extensive sweeps (multiple noise amplitudes, Lorenz gains, and random seeds), we observed a persistent decoupling between effective mass (m_eff) and control energy usage. Control effort stayed near baseline while m_eff either collapsed far below 1 or ballooned above 10, producing energy ratios that failed to scale with the inertia factor. This behavior satisfies the anomaly tripwire in AGENTS.md and warrants escalation.

## Key Run Conditions
- Controller: continuity-augmented PID with ≥5 salience channels, λ_c swept through [0, 10, 25, 50, 75, 100, 150, 200].
- Forcing:
  - Heavy Gaussian noise: σ ∈ {0.5, 0.3}.
  - Lorenz chaotic coupling: gain ∈ {0.0, 0.2}.
- Simulation horizon: 20 s, dt = 0.005.
- Seeds tested: 777, 1001, 31415, 4242, 99999, 2024.

## Evidence Snapshot
| Artifact | Forcing | Notable Observations |
| --- | --- | --- |
| `artifacts/mass_sweep/mass_stress_20251104_042444.json` | σ=0.5, gain=0.2, seed=777 | m_eff dropped to ≈0.026 while energy_ratio remained ≈1.00; control energy ≈77 vs. external energy ≈1600. |
| `artifacts/mass_sweep/mass_stress_20251104_043231.json` | σ=0.5, gain=0.2, seed=1001 | m_eff ranged 5–12 with energy_ratio ≈0.98; demonstrates runaway inertia despite steady control cost. |
| `artifacts/mass_sweep/mass_stress_20251104_043247.json` | σ=0.5, gain=0.2, seed=4242 | m_eff oscillated between 0.13 and 2.38 while energy_ratio hovered near unity. |
| `artifacts/mass_sweep/mass_stress_20251104_043302.json` | σ=0.5, gain=0.2, seed=99999 | m_eff spanned 0.67–9.0 with energy_ratio 0.94–1.02. |
| `artifacts/mass_sweep/mass_stress_20251104_042941.json` | σ=0.5, gain=0.0, seed=31415 | Noise-only forcing still yielded m_eff > 2 with energy_ratio ≈1.0. |
| `artifacts/mass_sweep/mass_stress_20251104_043518.json` | σ=0.3, gain=0.0, seed=2024 | Reduced noise kept the anomaly: m_eff up to 14 while energy_ratio ≈1.0 and external energy ≫ control energy. |

Representative metrics (averaged across λ>0 entries):
- Mean |m_eff − energy_ratio|: 0.54 → 5.59 depending on run.
- Mean energy_ratio / m_eff: 0.18 → 17.3 (significant divergence from unity).
- Salience variance ≈0.20, minimum salience capped at 0.001, indicating channel fatigue/surrender under strain.

## Interpretation
Continuity taxes appear to suppress active control while external forcing drives the plant, causing rise times to shrink or inflate independent of internal energy spending. This violates the expected proportionality between inertia (m_eff) and control energy outlined in Experiment P success criteria. The anomaly holds under multiple seeds and across different forcing configurations, making it unlikely to be stochastic noise.

## Immediate Actions
1. **Escalate** anomaly status (done via AGENTS.md update on 2025-11-04).
2. **Isolate contributors**: plan parameter sweeps varying noise σ and Lorenz gain separately to map boundaries of the effect.
3. **Cross-domain checks**: integrate findings into Experiment Q (coupled domain energy probe) to test whether energy can leak into auxiliary models under similar strain.

## Open Questions
- Does the anomaly persist with alternative plant dynamics (e.g., higher-order systems)?
- Can salience channel re-weighting or adaptive fatigue penalties restore proportional energy scaling?
- Is there a threshold of external energy where the controller fully disengages, effectively acting as an energy rectifier?
