# Salience Simulation Lab Notebook

This living document tracks experiments run inside the salience sandbox. It is not evidence of empirical phenomena—think of it as a lab diary for synthetic simulations. If script names or directories change, mirror those updates here so future reviewers can rebuild the same runs.

## High-Level Goals (Exploratory)
1. Observe how continuity-style taxes shape simulated inertial behaviour.
2. Explore whether continuity "strain" slows or enhances training in toy models.
3. Catalogue edge cases that look anomalous so they can be scrutinised, not to claim faster-than-light anything.
4. Map energy/intensity trade-offs for continuity-aware controllers in a controlled sandbox.

## SPARC Data Exploration (2025-11-05)
- Script: `empirical_validation/rotation_curve_validation.py`
- Purpose: Explore how the CSG V4 synthetic model performs when applied to real SPARC galaxy rotation curves as a fitting exercise (NOT validation).
- Usage: `python -m empirical_validation.rotation_curve_validation [--top-n N | --galaxy NAME ...]`
- Outputs: timestamped folders under `artifacts/empirical_validation/` containing JSON summaries plus the plots emitted by `CSGAnalysis`.
- Notes:
  - Defaults to the first 25 SPARC galaxies; repeat `--galaxy` flags override this selection.
  - Summary JSON reports best-fit κ₍c₎, bootstrap statistics, and aggregated residual metrics for quick inspection.
  - This is curve-fitting exploration, not empirical validation. The model is a heuristic sandbox, not a physics theory.

## CVG Lorenz Demo (2025-11-04)
- Script: `demos/cvg_lorenz/cvg_lorenz_demo.py`
- Purpose: produce the one-figure "CVG + OGY pulses" story (baseline PID vs. legacy multiplicative vs. modern additive gate) plus the two required ablations.
- Experimental observation in simulation: phase-timed causal pulses in the Lorenz controller use ≳10³× less energy than untimed control in this toy model.
- Outputs: `demos/cvg_lorenz/outputs/cvg_lorenz_demo.png` (figure) and timestamped JSON metrics in the same folder.
- Usage: `python -m demos.cvg_lorenz.cvg_lorenz_demo` from repo root (repro seed baked in).
- Ablations baked into run:
  1. Replace causal credit with shuffled exogenous surprises.
  2. Hold energy budget fixed but randomize pulse timing vs. OGY-aligned window.
- Implementation notes:
  - Uses `ModernSalienceParams` additive invariant (`scripts/salience_invariant_modern.py`).
  - Imports legacy invariant snapshot only for comparison; no core package wiring changed.
  - Lorenz plant is integrated by RK4; authority threshold shading appears on the figure.
  - JSON payload captures controller energies, authority, and ablation metrics for datasheet.

## Shared Vocabulary
- State x(t): latent controller or model state.
- Continuity tax: penalty on fast changes in high-salience components of x(t).
- Salience S' = DeltaA * R * M * (1 - phi).
- Continuity strain (sigma_tilde): cost density paid to absorb continuity-breaking operations.
- Effective mass factor m_eff = rise_time_90(lambda_c) / rise_time_90(lambda_c=0).
- Time dilation: extra steps needed to reach a target under continuity strain.

## Interpretation Notes (2025-11-04)
- The CSG V4 synthetic model computes a salience score S' from local continuity, retention, curvature, density, and delta-A terms, then transforms it to Q_local and Q_final as mathematical constructs (@src/csg_v4/model.py#32-88, @src/csg_v4/scores.py#17-164).
- The model's effective acceleration is scaled as a_eff = kappa_c * (a0 / Q_final), creating an interpolation between Newtonian and MOND-like regimes that produces flat synthetic rotation curves when fitted to data (@src/csg_v4/model.py#90-112). This is a curve-fitting exercise, not a claim about actual gravitational physics.
- Experimental scripts vary continuity taxes and salience gating to explore how increasing salience inertia in the model affects synthetic outputs, generating mass sweeps, adaptive inertia splits, and time-dilated learning regimes that follow the same continuity-strain mathematical framework used in the galaxy fits (@scripts/continuity_mass_sim.py#52-137, @scripts/unified_inertia_time_probe.py#83-162, @scripts/time_dilation_train.py#1-240).

## Experiment A: Continuity-Taxed Controller Mass Sweep
- Script: scripts/continuity_mass_sim.py
- Plant: 1D setpoint tracker, step 0 -> 1 at t=0, dt=0.01 s, horizon 5 s.
- Baseline PID: Kp=2.0, Ki=0.5, Kd=0.1.
- Sweep lambda_c in {0.00, 0.25, 0.50, 1.00, 2.00}.
- Metrics per lambda_c: rise_time_90, peak_overshoot, settling_time_2pct, rms_error, mean_salience, control_energy, derived m_eff.
- Log JSON rows to artifacts/mass_sweep/mass_sweep_<timestamp>.json.
- Success: m_eff increases smoothly with lambda_c while mean_salience stays >= 0.8.

## Experiment B: Adaptive Inertia Subspace
- Use same plant/controller as A.
- Split state into core (high tax) and edge (low tax) components.
- Example taxes: lambda_core = 2.0, lambda_edge = 0.1.
- Track lag_core, lag_edge, overshoot_core, overshoot_edge, m_eff_core, m_eff_edge, mean salience for each partition.
- Log to artifacts/mass_sweep/adaptive_mass_<timestamp>.json.
- Success: m_eff_core >> m_eff_edge while core salience stays high.

## Experiment C: Time Dilation via Continuity Strain
- Script: scripts/time_dilation_train.py
- Model: 2-layer MLP (64 hidden units) on simple classification task, fixed optimizer and batch size.
- Regime A: clean training baseline.
- Regime B: continuity-hostile training (contradictory labels every K steps, channel dropout, drift penalties).
- Metrics every 100 steps: validation accuracy/loss, mean S'. Compute t_converge (steps to hit target accuracy) for both regimes and time_dilation_factor = t_converge_B / t_converge_A.
- Log summary to artifacts/time_dilation/time_dilation_<timestamp>.json.
- Success: time_dilation_factor > 1.5 with S'_B > S'_A.

## Experiment D: Extended Causality Probe
- Script: scripts/causality_test_extended.py
- Prepare Bell pair; randomly apply slam (amplitude damping gamma=0.8) or quiet.
- Measure qubit B across bases {X, Y, Z} and time offsets {t0, t1, t2}. Target 400 trials x 500 shots.
- For each (basis, time_offset) bin compute p_slam, p_quiet, z-score, p-value, chi-square, trial count. Apply Bonferroni threshold p < 0.001.
- Log per-bin results to artifacts/causality_probe/causality_probe_extended_<timestamp>.json.
- Success: all bins null => no broadcast. Any stable significant bin => candidate causality leak.

## Experiment E: Energy Accounting / Proto-Landauer Check
- Use mass sweep and adaptive runs.
- Compare control energy ratios to m_eff.
- Log entries to artifacts/mass_sweep/mass_energy_<timestamp>.json with lambda values, m_eff, baseline energy, taxed energy, energy_ratio.
- Interpretation: ratio ~ m_eff (normal mass), ratio >> m_eff (continuity armor expensive), ratio << m_eff (potential thermodynamic anomaly; pause and replicate).

## Experiment F: Inertia–Time Coupling Probe
- Script: scripts/unified_inertia_time_probe.py
- Controller: adaptive core/edge policy tracking 1D plant as in Experiments A–B, but with gradient-based weight updates on a small replay buffer each step.
- Core state: continuity-taxed latent (lambda_core sweep) driving the main control channel.
- Subjective time dial: continuity strain (label noise, dropout, tax penalty) applied to the learning loop on a configurable cadence (strain_intensity sweep).
- Run grid over (lambda_core, strain_intensity) ∈ {(0, 0), (20, 0), (0, 1.0), (20, 1.0), (40, 1.5)}.
- Metrics per run: m_eff_core, convergence_steps (to |error| < 0.05 for 90% of window), learning_rate_effective (average weight delta magnitude), mean_salience_core, mean_salience_learning, coupling_ratio = convergence_steps / (baseline_steps * m_eff_core), Pearson(r) between control inertia trace and learning-rate trace, qualitative regime label (locked / decoupled / anomalous).
- Log rows to artifacts/unified_probe/unified_probe_<timestamp>.json with timestamps, config, metrics, run_id.
- Success: identify whether coupling_ratio ≈ 1 across grid (locked scalar) or discover decoupled regimes (ratio deviates >25% while salience stays >0.6). Escalate if regime is anomalous (ratio << 1 with low salience decay).

## Experiment G: Continuity Subsidy (Superfluid Mass Test)
- Script: scripts/continuity_subsidy_sim.py
- Plant/controller: reuse continuity_mass_sim plant, PID, dt=0.01 s, horizon 5 s.
- Introduce continuity subsidy coefficient μ_c that assists core acceleration when updates raise or preserve salience (ΔS' > 0 or S' ≥ 0.8) and reduce error magnitude.
- Sweep μ_c ∈ {0.0, 0.5, 1.0, 2.0} with λ_c fixed at 0.0.
- Metrics per μ_c: rise_time_90, peak_overshoot, settling_time_2pct, rms_error, mean_salience, control_energy, derived m_eff = rise_time_90(μ_c) / rise_time_90(μ_c=0).
- Success: decreasing m_eff with μ_c (ideally ≪ 1.0) while maintaining mean_salience ≥ 0.75 and avoiding large control_energy spikes.

## Experiment H: Burst Release / Stored Inertia Collapse
- Script: scripts/burst_release_sim.py
- Horizon 6 s, step target 0→1 at t=0. Phase 1 (0–1.5 s): high λ_core (e.g., 20) to accumulate continuity inertia; Phase 2: release by setting λ_core=0 and optionally enabling μ_c subsidy (0 or 2.0).
- Run two regimes: release-only (μ_c=0) and release-plus-subsidy (μ_c=2.0).
- Metrics: rise_time_90 measured post-release, post_release_peak_rate (max derivative within 0.5 s after release), post_release_overshoot (peak output within 0.5 s post-release minus target), mean_salience_phase2, control_energy_phase1 vs control_energy_phase2.
- Success: sharp drop in effective inertia post-release (rise_time_90 ↓, peak_rate ↑) while salience in phase 2 stays high; analyze energy redistribution between phases.

## Experiment I: Subjective Time Overclock (Learning Accelerator)
- Script: scripts/time_overclock_train.py
- Base on Experiment C architecture/dataset/optimizer.
- Regimes: baseline (as in Experiment C) and fast regime with salience-aware gradient gating every K steps (e.g., K=10). Channels with high ΔA, retention, payoff and low fatigue receive gradient boost α_boost; low-salience channels are attenuated.
- Sweep α_boost ∈ {1.0, 2.0, 4.0} (1.0 is control).
- Metrics: t_converge to fixed validation accuracy threshold, final_val_acc, mean_salience_latent, salience_active_fraction (fraction of steps where >20% of latent channels have salience > 0.7).
- Success: fast regime achieves t_converge < baseline while keeping salience metrics ≥ baseline.

## Experiment J: Reflex Mode (Zero-Lag Edge Channel)
- Script: scripts/reflex_mode_sim.py
- Start from adaptive_inertia_sim core/edge split. Keep core continuity nominal (λ_core≈0). Apply continuity subsidy μ_edge to edge channel rewarding salience-preserving, goal-aligned rapid updates.
- Sweep μ_edge ∈ {0.0, 1.0, 2.0, 5.0}.
- Metrics: edge_response_time_90, core_salience, cross_coupling_error (impact of edge motion on core stability), total overshoot.
- Success: edge response accelerates markedly while core salience remains ≥0.8 and cross_coupling_error stays low.

## Experiment K: Continuity Floor Gating (Morale Floor)
- Script: scripts/salience_floor_gate.py (helpers), integrated into fast-mode scripts.
- Salience floor S_FLOOR ∈ {0.6, 0.7, 0.8}. Any acceleration (subsidy μ_c, burst release assist, gradient boost α) is only applied if current salience S′ ≥ S_FLOOR.
- When S′ < S_FLOOR, acceleration is disabled and a recovery tax is applied (mild continuity penalty) to allow salience to recover.
- Metrics: mean_salience, min_salience, percentage of steps where acceleration was active vs blocked, plus context-specific metrics (effective mass, rise time, convergence steps, etc.).
- Success: salience never dips below the floor during fast phases; acceleration remains possible while maintaining high salience.

## Experiment L: Pulsed Overclock (Mastery Rhythm)
- Script: scripts/pulsed_overclock_train.py
- Training loop alternates between PULSE_ON_STEPS (boost) and PULSE_OFF_STEPS (cooldown). Boost applies only when salience exceeds floor; cooldown includes recovery tax.
- Parameters: PULSE_ON_STEPS=20, PULSE_OFF_STEPS=10, α_boost ∈ {2.0, 4.0}, S_FLOOR=0.7, recovery_tax ≈ 0.1–0.5.
- Metrics: t_converge (to val_acc ≥0.9 and ≥0.99), final accuracy/loss, mean/min/variance of salience, salience timeline, boost activation timeline, salience_active_fraction.
- Success: pulsed regime converges faster than baseline while maintaining salience above floor and showing appreciable boost activity.

## Experiment M: Soft Burst Release (Conscious Lunge)
- Script: scripts/soft_release_sim.py
- Two-phase controller: Phase1 high λ_core (lock), Phase2 ramps λ_core down and μ_c up over T_release. Salience floor gate blocks subsidy if S′ < S_FLOOR.
- Metrics: post-release peak rate, rise_time_90, mean/min salience in phase 2, energy stored in phase1 vs spent in phase2.
- Success: soft release achieves higher peak rate than release-only baseline while keeping salience_phase2_min ≥0.7 and avoiding extreme energy imbalance.

## Experiment N: Reflex Edge Controller (Low-Pass Merge)
- Script: scripts/reflex_edge_controller.py
- Split controller with fast edge (subsidized) and slow core (continuity-taxed). Edge acceleration gated by salience_edge; core refuses to merge when salience_core < floor.
- Low-pass merge: core_state ← core_state + MERGE_RATE*(edge_state - core_state) with MERGE_RATE ∈ {0.05, 0.1} when allowed.
- Sweep μ_edge ∈ {1.0, 2.0, 4.0}; λ_core fixed at stable baseline from Experiment B.
- Metrics: edge_response_time_90, core_response_time_90, min salience for core/edge, cross_coupling_error, identity_violation_flag.
- Success: edge responds quickly (<< baseline), core salience remains ≥0.8, no identity violations (core salience drop while merging), cross-coupling remains bounded.

## Experiment O: Fast Modes Summary
- Script: scripts/summarize_fast_modes.py
- Aggregates results from Experiments I–N (fast-mode variants) once completed.
- Metrics reported: fastest t_converge achieved (with config), highest peak_rate_post with salience_min ≥0.7, smallest edge_response_time_90 without identity violation, configurations maintaining mean salience ≥0.8 while exceeding baseline responsiveness/learning.
- Success: highlight regimes delivering “accelerated yet coherent” behavior.

## Experiment P: Extreme Continuity Stress Test
- Script: scripts/continuity_mass_sim.py (extended stress harness).
- Plant: baseline 1D tracker augmented with additive Gaussian noise (σ ≈ 0.1) and optional chaotic forcing (Lorenz-state coupling).
- Sweep λ_c ∈ {0.0, 5.0, 10.0, 20.0, 50.0}, expand salience vector (≥5 channels) to monitor component-specific fatigue.
- Metrics per λ_c: rise_time_90, settling_time_2pct, mean/min salience, control_energy, m_eff, energy_ratio (vs. baseline), salience variance, noise response energy.
- Success: detect regimes where energy_ratio ≪ 0.5 × m_eff or other paradoxical drops (continuity armor yielding apparent free energy). Flag and replicate immediately.

### Experiment P Status Summary (2025-11-04)
- Conducted multi-seed stress sweeps with heavy noise (σ=0.5) and Lorenz forcing gains up to 0.2. Control energy remained near baseline (~75–78) while effective mass collapsed to <0.2 or surged >10; energy_ratio persisted in a narrow 0.94–1.03 band, confirming decoupling between m_eff and energy expenditure.
- Noise-only sweeps (Lorenz gain 0.0) sustained the energy–mass mismatch: m_eff rose to 2.8–14 while energy_ratio stayed ≈1.0. Even reduced noise (σ=0.3) produced external energy orders of magnitude larger than control effort and preserved the anomaly.
- Quantitative diagnostics across artifacts showed mean |m_eff − energy_ratio| ranging 0.54–5.6 and mean (energy_ratio / m_eff) spanning 0.18–17.3. Result: Within this toy simulation, programmable inertia diverges from energy scaling under continuity strain; unexpected behavior flagged for further investigation.
- Added control-authority diagnostics (control_energy vs. external energy through the rise window) in scripts/continuity_mass_sim.py. Under heavy forcing every run failed the authority gate (ratios ≈0.06–0.11), so new m_eff and energy_ratio values are withheld until the loop regains control; this reframes the earlier anomaly as a control-failure artifact under these conditions.
- Next steps: tighten salience floors or reduce forcing so at least one taxed regime passes the authority gate; rerun to test whether genuine continuity-taxed behavior still breaks energy scaling once the controller is in charge.
- First zero-noise rerun confirms the gate works: λ_c ∈ {0,5,10} retain authority (ratios → ∞), with m_eff rising to 2.2–3.5 while energy_ratio slips modestly below 1 because the taxed controller expends less energy over the fixed horizon before reaching 90%. Higher λ_c fails to hit the rise threshold within the horizon, so m_eff/energy_ratio remain NaN. Need longer horizons or per-error energy normalization to decide whether any residual anomaly remains when authority holds.
- Breathing-stress probe: added cyclic noise modulation (inhale σ scaled to 0.1 for 4 s, exhale full σ for 2 s) with noise delay. Breathing run maintained control authority and trimmed rms_error by ≈12% while conserving energy_ratio ≈0.65–1.0 across λ_c, suggesting structured low-noise windows help salience recover before stress pulses—metaphorically “take a deep breath” before turbulence.
- Lorenz + breathing: enabling chaotic forcing (gain 0.2) during the exhale phase overwhelmed the controller; authority ratios fell to 0.49–0.70 despite the quiet inhale, so m_eff/energy_ratio were withheld. Next: strengthen the recovery window (longer inhale, deeper σ cut, or staged Lorenz ramp) and/or add adaptive gating that pauses chaotic forcing until authority > 1 to prevent false anomalies.
- Lorenz + adaptive breathing: adaptive schedule reacted by driving the disturbance scale toward the minimum (mean ≈0.019, min ≈0.005) but authority still stayed <1 (0.49–0.70), mirroring the static breathing failure @artifacts/mass_sweep/mass_stress_20251104_064818.json#1-137. Suggests chaotic forcing needs explicit gating or staged ramp rather than passive breathing; next run will combine adaptive breathing with Lorenz pause until authority > 1 and evaluate re-entry protocols.
- Lorenz gate + adaptive breathing: adding authority-gated Lorenz forcing kept the chaotic channel off roughly 70% of the time (gate_active_mean ≈0.25–0.32) until authority recovered, yet authority never exceeded ~0.32 and the gate rarely reopened @artifacts/mass_sweep/mass_stress_20251104_065516.json#1-137. Conclusion: once chaotic forcing overwhelms authority, passive gating simply leaves it off; need ramped re-entry (e.g., proportional gain, staged pulses) or predictive engagement to restore coherence.
- Pulse probes (micro vs. macro rest schedules): 10 micro rest pulses per 6 s breath (0.2 duty) kept the Lorenz gate mostly shut but authority still stalled at 0.18–0.20 @artifacts/mass_sweep/mass_stress_20251104_070733.json#1-84; a long 60 s inhale (horizon 120 s) raised breathing averages but authority plateaued ≈0.31 with chaos reasserting control @artifacts/mass_sweep/mass_stress_20251104_070850.json#1-84. Rest windows alone can’t win back the ring—need coordinated re-entry cues keyed to authority slope or predictive counters.
- Hot-tag pulses (authority-slope triggers): enabling slope-based pulses never fired a single tag (trigger_count=0) because authority ratios never stayed above the gate long enough; Lorenz forcing stayed mostly disabled (scale_mean ≈0.04–0.08) and chaos still dominated @artifacts/mass_sweep/mass_stress_20251104_072618.json#1-84. Need lower slope thresholds or predictive partner cues, possibly combining with Experiment Q to deliver the scripted comeback.
- Hot-tag heat + near-tag tracker: expanded logic tracks heat length, near-tag attempts, and control boost during tag windows, but first run still logged zero activations even after 400-step heat, 399 near tags, and gates held shut @artifacts/mass_sweep/mass_stress_20251104_074816.json#1-84. Next: relax authority gate post-heat, introduce explicit partner controller swap, or queue Experiment Q ally to seize the legal tag.
- Baseline (ban infinite): disabling Lorenz chaos immediately restored authority ≥ 1.0, salience rebounded to 0.70–0.92, and m_eff re-aligned with energy_ratio across λ_c ∈ {0,5,10} @artifacts/mass_sweep/mass_stress_20251104_083145.json#1-64 and @artifacts/mass_sweep/mass_stress_20251104_083153.json#1-64. Confirms the Experiment P unexpected behavior is an artifact of control failure when chaos forcing is removed.
- Mirror assist (auto hot tag + subsidy): preloading a partner with assist gain=5 and Lorenz still active forced authority ≥1 but only by injecting massive control/assist energy (m_eff ≈1, energy_ratio ≫1, salience drifting toward floor) while Lorenz gating stayed almost fully engaged @artifacts/mass_sweep/mass_stress_20251104_084250.json#1-84. Needs sustainable assist (e.g., Experiment Q coupling) that breaks the combo without brute-force energy spikes.

## Experiment Q: Coupled Domain Energy Leak Probe
- Script: scripts/coupled_domain_energy.py (new).
- System: continuity-taxed PID controller coupled bidirectionally with a small MLP predictor (shared λ_c on control + latent weights).
- Regimes: control-only baseline, forward-coupled (ML predicts setpoint), reverse-coupled (control state feeds ML), bidirectional loop.
- Sweep λ_c ∈ {0.0, 1.0, 5.0}; evaluate optional subsidy μ_shared ∈ {0, 1.0}.
- Metrics: total control_energy, ML update energy (‖Δw‖₁), combined energy_ratio vs. isolated baselines, m_eff_combined, cross-domain salience correlation.
- Success: any coupled regime showing energy_ratio significantly below additive expectation (potential energy leak or synergy anomaly).

## Experiment R: Temporal Energy Decay Probe
- Script: scripts/temporal_decay_probe.py (new).
- Setup: long-horizon plant (T = 20 s, dt = 0.01) with exponential fatigue amplification and scheduled perturbations.
- Sweep λ_c ∈ {0.0, 2.0, 10.0}; track metrics in early/mid/late windows (e.g., 0–5 s, 5–10 s, 10–20 s).
- Metrics: windowed energy_ratio, windowed m_eff, cumulative fatigue, salience drift rate, entropy proxies.
- Success: late-window energy_ratio ≪ early-window ratio while salience remains ≥0.7, suggesting temporal energy anomalies or storage effects.

## Experiment S: Neuro-Salience Continuity Trial
- Script: scripts/neuro_salience_trial.py (new).
- Model: spiking neural network (e.g., Izhikevich) performing pattern classification with continuity taxes on synaptic updates.
- Sweep λ_c ∈ {0.0, 0.5, 2.0}; vary fatigue gain to mimic cognitive load.
- Metrics: classification accuracy, learning speed, synaptic energy (Σ|Δw|), salience vs. firing rate correlations, effective inertia in response times.
- Success: observe cognitive-style slowdowns (higher inertia) with potential energy anomalies or robustness improvements under strain.

## Experiment T: Quantum-Classical Salience Entanglement
- Script: scripts/quantum_salience_hybrid.py (new).
- Extend Bell-pair probe by mapping classical salience states to measurement bases (e.g., high salience → X basis, low salience → Z).
- Runs: ≥1000 trials × 1000 shots across {X, Y, Z} bases and offsets; log salience-conditioned probabilities.
- Metrics: p_slam vs. salience bin, z-score per bin, mutual information between salience and quantum outcomes, Bonferroni-corrected significance.
- Success: any statistically significant deviation conditioned on salience (candidate hybrid retrocausality). Escalate immediately.

## Experiment U: Cluster-Scale Salience Coupling
- Script: scripts/cluster_salience_fit.py (new).
- Data: extend SPARC fits to galaxy clusters (e.g., Virgo subset) with shared κ_c and inter-galaxy continuity constraints.
- Approach: fit galaxies jointly with salience coupling penalties encouraging coherent κ_c drift.
- Metrics: per-galaxy |dv|/v, κ_c variance, cluster-level salience covariance, energy accounting across cluster runs.
- Success: discover anomalous κ_c correlations or improved fits beyond independent models, suggesting large-scale continuity effects.

## Experiment V: Adversarial Salience Attack
- Script: scripts/adversarial_salience_attack.py (new).
- Baseline: time_overclock_train architecture with and without continuity taxes (λ_c = 0, 1.0).
- Procedure: craft adversarial perturbations targeting lowest-salience channels every K steps; compare robustness vs. standard PGD/FGSM.
- Metrics: accuracy drop under attack, salience recovery time, energy spent defending (added control/gradient magnitude), m_eff change post-attack.
- Success: continuity regime exhibits anomalously low energy cost for attack resilience or reveals failure modes where salience gating collapses.

## Logging Requirements
- Every JSON row must include timestamp, experiment_name, parameter settings, metrics, run_id or trial_id if relevant.
- Prefer UTC timestamps.
- Maintain reproducible seeds and document overrides.

## Interpretation Tripwires
1. Programmable inertia: monotonic m_eff vs lambda_c with high salience.
2. Directional inertia: m_eff_core >> m_eff_edge.
3. Time dilation: time_dilation_factor > 1.5 with higher salience in strained regime.
4. No-signaling closure: extended probe null results or flagged candidate for hardware.
5. Energy scaling anomaly: energy_ratio << m_eff triggers replication.

## Execution Order
1. Experiment A (mass sweep).
2. Experiment B (adaptive inertia).
3. Experiment C (time dilation).
4. Experiment D (extended causality probe).
5. Experiment E (energy scaling).

If experiments D or E show anomalies, escalate immediately for deeper investigation.
