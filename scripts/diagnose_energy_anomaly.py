"""Diagnostic script to investigate the energy-mass mismatch in Experiment E.

This script analyzes why the adaptive regime shows m_eff_core = 1.413 (41% more inertia)
but energy_ratio = 0.203 (79% less energy), a 6.95× mismatch.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import the adaptive controller
import sys
sys.path.insert(0, str(Path(__file__).parent))
from adaptive_inertia_sim import (
    AdaptiveController,
    ContinuitySettings,
    ExperimentConfig,
    simulate,
    compute_metrics,
)

ARTIFACT_DIR = Path("artifacts/energy_anomaly_investigation")


def simulate_with_traces(
    settings: ContinuitySettings, cfg: ExperimentConfig
) -> Dict[str, np.ndarray]:
    """Run simulation and return detailed traces for analysis."""
    controller = AdaptiveController(cfg, settings)
    controller.reset()

    steps = int(cfg.horizon / cfg.dt)
    target = np.ones(steps)
    output = np.zeros(steps)
    core_trace = np.zeros(steps)
    edge_trace = np.zeros(steps)
    salience_core = np.zeros(steps)
    salience_edge = np.zeros(steps)
    control_trace = np.zeros(steps)
    control_energy_trace = np.zeros(steps)
    core_state_trace = np.zeros(steps)
    edge_state_trace = np.zeros(steps)
    error_trace = np.zeros(steps)

    control_energy = 0.0
    state = 0.0

    for i in range(steps):
        error = target[i] - state
        error_trace[i] = error
        result = controller.step(cfg.dt, error)
        state += cfg.dt * ((-state + result["control"]) / cfg.tau)

        output[i] = state
        core_trace[i] = result["core_output"]
        edge_trace[i] = result["edge_output"]
        control_trace[i] = result["control"]
        salience_core[i] = result["salience_core"]
        salience_edge[i] = result["salience_edge"]
        core_state_trace[i] = controller.core_state
        edge_state_trace[i] = controller.edge_state

        energy_step = abs(result["control"]) * cfg.dt
        control_energy += energy_step
        control_energy_trace[i] = energy_step

    return {
        "time": np.arange(0.0, cfg.horizon, cfg.dt),
        "output": output,
        "core": core_trace,
        "edge": edge_trace,
        "control": control_trace,
        "control_energy_trace": control_energy_trace,
        "control_energy": control_energy,
        "salience_core": salience_core,
        "salience_edge": salience_edge,
        "core_state": core_state_trace,
        "edge_state": edge_state_trace,
        "error": error_trace,
    }


def compute_component_wise_metrics(
    baseline_traces: Dict[str, np.ndarray],
    adaptive_traces: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
) -> Dict[str, float]:
    """Compute separate m_eff for core and edge based on their individual contributions."""

    def compute_component_lag(component_output: np.ndarray, time: np.ndarray, dt: float) -> float:
        """Compute the lag (time to 50% of total area) for a component."""
        abs_output = np.abs(component_output)
        total_area = float(np.sum(abs_output) * dt)
        if total_area < 1e-6:
            return float('nan')
        cumulative = np.cumsum(abs_output) * dt
        target = 0.5 * total_area
        idx = int(np.searchsorted(cumulative, target))
        return float(time[min(idx, len(time) - 1)])

    # Compute lags for each component
    baseline_core_lag = compute_component_lag(baseline_traces["core"], baseline_traces["time"], cfg.dt)
    adaptive_core_lag = compute_component_lag(adaptive_traces["core"], adaptive_traces["time"], cfg.dt)

    baseline_edge_lag = compute_component_lag(baseline_traces["edge"], baseline_traces["time"], cfg.dt)
    adaptive_edge_lag = compute_component_lag(adaptive_traces["edge"], adaptive_traces["time"], cfg.dt)

    # Compute component energy contributions
    baseline_core_energy = float(np.sum(np.abs(baseline_traces["core"])) * cfg.dt)
    adaptive_core_energy = float(np.sum(np.abs(adaptive_traces["core"])) * cfg.dt)

    baseline_edge_energy = float(np.sum(np.abs(baseline_traces["edge"])) * cfg.dt)
    adaptive_edge_energy = float(np.sum(np.abs(adaptive_traces["edge"])) * cfg.dt)

    # Compute m_eff for each component
    m_eff_core = adaptive_core_lag / baseline_core_lag if baseline_core_lag > 0 else float('nan')
    m_eff_edge = adaptive_edge_lag / baseline_edge_lag if baseline_edge_lag > 0 else float('nan')

    # Compute energy ratios for each component
    energy_ratio_core = adaptive_core_energy / baseline_core_energy if baseline_core_energy > 0 else float('nan')
    energy_ratio_edge = adaptive_edge_energy / baseline_edge_energy if baseline_edge_energy > 0 else float('nan')

    # Compute total system rise time (90% of target)
    def compute_rise_time_90(output: np.ndarray, time: np.ndarray) -> float:
        threshold = 0.9
        indices = np.where(output >= threshold)[0]
        if indices.size == 0:
            return float('nan')
        return float(time[indices[0]])

    baseline_rise = compute_rise_time_90(baseline_traces["output"], baseline_traces["time"])
    adaptive_rise = compute_rise_time_90(adaptive_traces["output"], adaptive_traces["time"])

    return {
        "baseline_core_lag": baseline_core_lag,
        "adaptive_core_lag": adaptive_core_lag,
        "m_eff_core": m_eff_core,
        "baseline_core_energy": baseline_core_energy,
        "adaptive_core_energy": adaptive_core_energy,
        "energy_ratio_core": energy_ratio_core,
        "baseline_edge_lag": baseline_edge_lag,
        "adaptive_edge_lag": adaptive_edge_lag,
        "m_eff_edge": m_eff_edge,
        "baseline_edge_energy": baseline_edge_energy,
        "adaptive_edge_energy": adaptive_edge_energy,
        "energy_ratio_edge": energy_ratio_edge,
        "baseline_rise_time": baseline_rise,
        "adaptive_rise_time": adaptive_rise,
        "system_m_eff": adaptive_rise / baseline_rise if baseline_rise > 0 else float('nan'),
        "baseline_total_energy": baseline_traces["control_energy"],
        "adaptive_total_energy": adaptive_traces["control_energy"],
        "system_energy_ratio": adaptive_traces["control_energy"] / baseline_traces["control_energy"],
    }


def analyze_energy_distribution(
    baseline_traces: Dict[str, np.ndarray],
    adaptive_traces: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
) -> Dict[str, any]:
    """Analyze where the energy is spent in each regime."""

    # Find the 90% rise time index for each
    def find_rise_idx(output: np.ndarray) -> int:
        indices = np.where(output >= 0.9)[0]
        return int(indices[0]) if indices.size > 0 else len(output) - 1

    baseline_rise_idx = find_rise_idx(baseline_traces["output"])
    adaptive_rise_idx = find_rise_idx(adaptive_traces["output"])

    # Energy spent before rise time
    baseline_energy_before_rise = float(np.sum(baseline_traces["control_energy_trace"][:baseline_rise_idx+1]))
    adaptive_energy_before_rise = float(np.sum(adaptive_traces["control_energy_trace"][:adaptive_rise_idx+1]))

    # Energy spent after rise time
    baseline_energy_after_rise = float(np.sum(baseline_traces["control_energy_trace"][baseline_rise_idx+1:]))
    adaptive_energy_after_rise = float(np.sum(adaptive_traces["control_energy_trace"][adaptive_rise_idx+1:]))

    # Peak control effort
    baseline_peak_control = float(np.max(np.abs(baseline_traces["control"])))
    adaptive_peak_control = float(np.max(np.abs(adaptive_traces["control"])))

    # Average control effort during rise
    baseline_avg_control = float(np.mean(np.abs(baseline_traces["control"][:baseline_rise_idx+1])))
    adaptive_avg_control = float(np.mean(np.abs(adaptive_traces["control"][:adaptive_rise_idx+1])))

    # Component contributions at peak
    baseline_peak_idx = int(np.argmax(np.abs(baseline_traces["control"])))
    adaptive_peak_idx = int(np.argmax(np.abs(adaptive_traces["control"])))

    return {
        "baseline_rise_idx": baseline_rise_idx,
        "adaptive_rise_idx": adaptive_rise_idx,
        "baseline_rise_time_steps": baseline_rise_idx,
        "adaptive_rise_time_steps": adaptive_rise_idx,
        "baseline_energy_before_rise": baseline_energy_before_rise,
        "adaptive_energy_before_rise": adaptive_energy_before_rise,
        "baseline_energy_after_rise": baseline_energy_after_rise,
        "adaptive_energy_after_rise": adaptive_energy_after_rise,
        "energy_before_rise_ratio": adaptive_energy_before_rise / baseline_energy_before_rise,
        "energy_after_rise_ratio": adaptive_energy_after_rise / baseline_energy_after_rise if baseline_energy_after_rise > 0 else float('nan'),
        "baseline_peak_control": baseline_peak_control,
        "adaptive_peak_control": adaptive_peak_control,
        "peak_control_ratio": adaptive_peak_control / baseline_peak_control,
        "baseline_avg_control": baseline_avg_control,
        "adaptive_avg_control": adaptive_avg_control,
        "avg_control_ratio": adaptive_avg_control / baseline_avg_control,
        "baseline_peak_core_contrib": float(baseline_traces["core"][baseline_peak_idx]),
        "baseline_peak_edge_contrib": float(baseline_traces["edge"][baseline_peak_idx]),
        "adaptive_peak_core_contrib": float(adaptive_traces["core"][adaptive_peak_idx]),
        "adaptive_peak_edge_contrib": float(adaptive_traces["edge"][adaptive_peak_idx]),
    }


def plot_diagnostic_figures(
    baseline_traces: Dict[str, np.ndarray],
    adaptive_traces: Dict[str, np.ndarray],
    component_metrics: Dict[str, float],
    energy_dist: Dict[str, any],
    output_dir: Path,
) -> None:
    """Create comprehensive diagnostic plots."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Energy-Mass Mismatch Diagnostic Analysis', fontsize=16, fontweight='bold')

    # Row 1: System Response
    # Plot 1: Output traces
    ax = axes[0, 0]
    ax.plot(baseline_traces["time"], baseline_traces["output"], 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["output"], 'r-', label='Adaptive (λ=40)', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target')
    ax.axhline(y=0.9, color='k', linestyle=':', alpha=0.3, label='90% threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('System Output')
    ax.set_title('System Response Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Control effort over time
    ax = axes[0, 1]
    ax.plot(baseline_traces["time"], baseline_traces["control"], 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["control"], 'r-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Signal')
    ax.set_title('Control Effort Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative energy
    ax = axes[0, 2]
    baseline_cumulative = np.cumsum(baseline_traces["control_energy_trace"])
    adaptive_cumulative = np.cumsum(adaptive_traces["control_energy_trace"])
    ax.plot(baseline_traces["time"], baseline_cumulative, 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_cumulative, 'r-', label='Adaptive', linewidth=2)

    # Mark rise times
    baseline_rise_idx = energy_dist["baseline_rise_idx"]
    adaptive_rise_idx = energy_dist["adaptive_rise_idx"]
    ax.axvline(x=baseline_traces["time"][baseline_rise_idx], color='b', linestyle='--', alpha=0.5, label='Baseline rise')
    ax.axvline(x=adaptive_traces["time"][adaptive_rise_idx], color='r', linestyle='--', alpha=0.5, label='Adaptive rise')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Energy')
    ax.set_title('Cumulative Control Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Component Analysis
    # Plot 4: Core component output
    ax = axes[1, 0]
    ax.plot(baseline_traces["time"], baseline_traces["core"], 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["core"], 'r-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Core Output')
    ax.set_title(f'Core Component (m_eff={component_metrics["m_eff_core"]:.3f}, E_ratio={component_metrics["energy_ratio_core"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Edge component output
    ax = axes[1, 1]
    ax.plot(baseline_traces["time"], baseline_traces["edge"], 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["edge"], 'r-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Edge Output')
    ax.set_title(f'Edge Component (m_eff={component_metrics["m_eff_edge"]:.3f}, E_ratio={component_metrics["energy_ratio_edge"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Component contributions as stacked area
    ax = axes[1, 2]
    ax.fill_between(baseline_traces["time"], 0, np.abs(baseline_traces["core"]),
                     alpha=0.5, label='Baseline Core', color='lightblue')
    ax.fill_between(baseline_traces["time"], np.abs(baseline_traces["core"]),
                     np.abs(baseline_traces["core"]) + np.abs(baseline_traces["edge"]),
                     alpha=0.5, label='Baseline Edge', color='lightgreen')

    offset = max(np.abs(baseline_traces["core"]) + np.abs(baseline_traces["edge"])) * 1.2
    ax.fill_between(adaptive_traces["time"], offset, offset + np.abs(adaptive_traces["core"]),
                     alpha=0.5, label='Adaptive Core', color='salmon')
    ax.fill_between(adaptive_traces["time"], offset + np.abs(adaptive_traces["core"]),
                     offset + np.abs(adaptive_traces["core"]) + np.abs(adaptive_traces["edge"]),
                     alpha=0.5, label='Adaptive Edge', color='lightcoral')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Component Magnitude')
    ax.set_title('Component Contributions (Stacked)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: Salience and States
    # Plot 7: Salience evolution
    ax = axes[2, 0]
    ax.plot(baseline_traces["time"], baseline_traces["salience_core"], 'b-', label='Baseline Core', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["salience_core"], 'r-', label='Adaptive Core', linewidth=2)
    ax.plot(baseline_traces["time"], baseline_traces["salience_edge"], 'b--', label='Baseline Edge', linewidth=1)
    ax.plot(adaptive_traces["time"], adaptive_traces["salience_edge"], 'r--', label='Adaptive Edge', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Salience')
    ax.set_title('Salience Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 8: Internal states
    ax = axes[2, 1]
    ax.plot(baseline_traces["time"], baseline_traces["core_state"], 'b-', label='Baseline Core State', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["core_state"], 'r-', label='Adaptive Core State', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Core State')
    ax.set_title('Core Internal State Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 9: Error signal
    ax = axes[2, 2]
    ax.plot(baseline_traces["time"], baseline_traces["error"], 'b-', label='Baseline', linewidth=2)
    ax.plot(adaptive_traces["time"], adaptive_traces["error"], 'r-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error')
    ax.set_title('Error Signal (Target - Output)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "diagnostic_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plots saved to {output_path}")
    plt.close()


def generate_analysis_report(
    baseline_traces: Dict[str, np.ndarray],
    adaptive_traces: Dict[str, np.ndarray],
    component_metrics: Dict[str, float],
    energy_dist: Dict[str, any],
    cfg: ExperimentConfig,
    baseline_settings: ContinuitySettings,
    adaptive_settings: ContinuitySettings,
) -> Dict:
    """Generate comprehensive analysis report."""

    # Calculate mismatch ratio
    system_m_eff = component_metrics["system_m_eff"]
    system_energy_ratio = component_metrics["system_energy_ratio"]
    mismatch_ratio = system_m_eff / system_energy_ratio if system_energy_ratio > 0 else float('nan')

    # Determine the likely explanation
    explanations = []
    confidence_scores = {}

    # Hypothesis 1: Architectural efficiency (adaptive reaches target with less oscillation)
    overshoot_reduction = (energy_dist["baseline_energy_after_rise"] - energy_dist["adaptive_energy_after_rise"]) / baseline_traces["control_energy"]
    if overshoot_reduction > 0.3:  # More than 30% of baseline energy saved post-rise
        explanations.append("ARCHITECTURAL_EFFICIENCY")
        confidence_scores["ARCHITECTURAL_EFFICIENCY"] = min(overshoot_reduction * 100, 100)

    # Hypothesis 2: Peak control reduction (slower rise but gentler control)
    peak_ratio = energy_dist["peak_control_ratio"]
    avg_ratio = energy_dist["avg_control_ratio"]
    if peak_ratio < 0.5 and avg_ratio < 0.5:
        explanations.append("GENTLER_CONTROL_TRAJECTORY")
        confidence_scores["GENTLER_CONTROL_TRAJECTORY"] = (1.0 - peak_ratio) * 100

    # Hypothesis 3: Component rebalancing (edge does derivative work more efficiently)
    core_energy_savings = 1.0 - component_metrics["energy_ratio_core"]
    edge_energy_savings = 1.0 - component_metrics["energy_ratio_edge"]
    if core_energy_savings > 0.5:  # Core uses 50% less
        explanations.append("CORE_COMPONENT_EFFICIENCY")
        confidence_scores["CORE_COMPONENT_EFFICIENCY"] = core_energy_savings * 100

    # Hypothesis 4: Metric definition problem (m_eff measures wrong thing)
    # If core lag increases but system rise time doesn't increase proportionally
    core_lag_increase = (component_metrics["adaptive_core_lag"] - component_metrics["baseline_core_lag"]) / component_metrics["baseline_core_lag"]
    system_rise_increase = (component_metrics["adaptive_rise_time"] - component_metrics["baseline_rise_time"]) / component_metrics["baseline_rise_time"]
    if core_lag_increase > 0.3 and system_rise_increase < 0.1:
        explanations.append("METRIC_DEFINITION_MISMATCH")
        confidence_scores["METRIC_DEFINITION_MISMATCH"] = abs(core_lag_increase - system_rise_increase) * 100

    # Hypothesis 5: Continuity armor (salience-driven mass actually improves efficiency)
    salience_core_adaptive = float(np.mean(adaptive_traces["salience_core"]))
    salience_core_baseline = float(np.mean(baseline_traces["salience_core"]))
    if salience_core_adaptive < salience_core_baseline * 0.95:
        explanations.append("CONTINUITY_ARMOR_EFFECT")
        confidence_scores["CONTINUITY_ARMOR_EFFECT"] = (1.0 - salience_core_adaptive / salience_core_baseline) * 100

    # Determine primary explanation
    if confidence_scores:
        primary_explanation = max(confidence_scores.items(), key=lambda x: x[1])
    else:
        primary_explanation = ("UNKNOWN", 0.0)

    report = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "experiment": "experiment_e_energy_mass_mismatch",
            "baseline_lambda_core": baseline_settings.lambda_core,
            "baseline_lambda_edge": baseline_settings.lambda_edge,
            "adaptive_lambda_core": adaptive_settings.lambda_core,
            "adaptive_lambda_edge": adaptive_settings.lambda_edge,
        },
        "anomaly_summary": {
            "system_m_eff": float(system_m_eff),
            "system_energy_ratio": float(system_energy_ratio),
            "mismatch_ratio": float(mismatch_ratio),
            "description": f"System shows {(system_m_eff-1)*100:.1f}% more inertia but uses {(1-system_energy_ratio)*100:.1f}% less energy",
            "mismatch_factor": f"{mismatch_ratio:.2f}×"
        },
        "component_analysis": {
            "core": {
                "m_eff": float(component_metrics["m_eff_core"]),
                "energy_ratio": float(component_metrics["energy_ratio_core"]),
                "component_mismatch": float(component_metrics["m_eff_core"] / component_metrics["energy_ratio_core"]),
                "baseline_lag": float(component_metrics["baseline_core_lag"]),
                "adaptive_lag": float(component_metrics["adaptive_core_lag"]),
                "baseline_energy": float(component_metrics["baseline_core_energy"]),
                "adaptive_energy": float(component_metrics["adaptive_core_energy"]),
            },
            "edge": {
                "m_eff": float(component_metrics["m_eff_edge"]),
                "energy_ratio": float(component_metrics["energy_ratio_edge"]),
                "component_mismatch": float(component_metrics["m_eff_edge"] / component_metrics["energy_ratio_edge"]) if component_metrics["energy_ratio_edge"] > 0 else float('nan'),
                "baseline_lag": float(component_metrics["baseline_edge_lag"]),
                "adaptive_lag": float(component_metrics["adaptive_edge_lag"]),
                "baseline_energy": float(component_metrics["baseline_edge_energy"]),
                "adaptive_energy": float(component_metrics["adaptive_edge_energy"]),
            },
        },
        "energy_distribution": {
            "baseline_rise_time_s": float(component_metrics["baseline_rise_time"]),
            "adaptive_rise_time_s": float(component_metrics["adaptive_rise_time"]),
            "baseline_total_energy": float(component_metrics["baseline_total_energy"]),
            "adaptive_total_energy": float(component_metrics["adaptive_total_energy"]),
            "baseline_energy_before_rise": float(energy_dist["baseline_energy_before_rise"]),
            "adaptive_energy_before_rise": float(energy_dist["adaptive_energy_before_rise"]),
            "baseline_energy_after_rise": float(energy_dist["baseline_energy_after_rise"]),
            "adaptive_energy_after_rise": float(energy_dist["adaptive_energy_after_rise"]),
            "energy_before_rise_ratio": float(energy_dist["energy_before_rise_ratio"]),
            "energy_after_rise_ratio": float(energy_dist["energy_after_rise_ratio"]),
            "baseline_peak_control": float(energy_dist["baseline_peak_control"]),
            "adaptive_peak_control": float(energy_dist["adaptive_peak_control"]),
            "peak_control_ratio": float(energy_dist["peak_control_ratio"]),
            "baseline_avg_control": float(energy_dist["baseline_avg_control"]),
            "adaptive_avg_control": float(energy_dist["adaptive_avg_control"]),
            "avg_control_ratio": float(energy_dist["avg_control_ratio"]),
        },
        "diagnostic_findings": {
            "overshoot_energy_fraction": float(energy_dist["baseline_energy_after_rise"] / baseline_traces["control_energy"]),
            "adaptive_overshoot_energy_fraction": float(energy_dist["adaptive_energy_after_rise"] / adaptive_traces["control_energy"]),
            "overshoot_savings_as_pct_of_total": float(overshoot_reduction * 100),
            "core_lag_vs_system_rise_discrepancy": float(abs(core_lag_increase - system_rise_increase)) if not math.isnan(system_rise_increase) else float('nan'),
            "salience_core_mean_baseline": float(np.mean(baseline_traces["salience_core"])),
            "salience_core_mean_adaptive": float(np.mean(adaptive_traces["salience_core"])),
            "salience_reduction_pct": float((1.0 - np.mean(adaptive_traces["salience_core"]) / np.mean(baseline_traces["salience_core"])) * 100),
        },
        "hypotheses": {
            "candidate_explanations": explanations,
            "confidence_scores": {k: float(v) for k, v in confidence_scores.items()},
            "primary_explanation": primary_explanation[0],
            "primary_confidence": float(primary_explanation[1]),
        },
        "conclusions": {
            "is_real_effect": len(explanations) > 0,
            "is_measurement_artifact": "METRIC_DEFINITION_MISMATCH" in explanations,
            "is_architectural_efficiency": "ARCHITECTURAL_EFFICIENCY" in explanations or "GENTLER_CONTROL_TRAJECTORY" in explanations,
            "is_continuity_armor": "CONTINUITY_ARMOR_EFFECT" in explanations or "CORE_COMPONENT_EFFICIENCY" in explanations,
            "verdict": _generate_verdict(explanations, confidence_scores, mismatch_ratio),
        },
    }

    return report


def _generate_verdict(explanations: List[str], confidence_scores: Dict[str, float], mismatch_ratio: float) -> str:
    """Generate a human-readable verdict on the anomaly."""

    if not explanations:
        return (
            f"UNKNOWN: The {mismatch_ratio:.2f}× energy-mass mismatch cannot be explained by standard "
            "hypotheses. Further investigation required."
        )

    if "METRIC_DEFINITION_MISMATCH" in explanations and confidence_scores.get("METRIC_DEFINITION_MISMATCH", 0) > 50:
        return (
            "METRIC ARTIFACT: The mismatch is primarily due to measuring different aspects of system behavior. "
            "The 'm_eff' metric (based on component lag) doesn't directly correspond to the energy metric "
            "(based on total control effort). The 'effective mass' metaphor is misleading here."
        )

    if "ARCHITECTURAL_EFFICIENCY" in explanations and confidence_scores.get("ARCHITECTURAL_EFFICIENCY", 0) > 50:
        return (
            f"ARCHITECTURAL EFFICIENCY: The adaptive system is fundamentally more efficient. The {mismatch_ratio:.2f}× "
            "mismatch is genuine - the adaptive architecture reaches the target more slowly but with dramatically "
            "less overshoot and oscillation, saving significant energy. The continuity tax on the core component "
            "acts as a beneficial damper, preventing wasteful control effort."
        )

    if "GENTLER_CONTROL_TRAJECTORY" in explanations:
        return (
            "GENTLER TRAJECTORY: The continuity tax forces the controller to take a more gradual, energy-efficient "
            f"path to the target. While this increases rise time ({mismatch_ratio:.2f}× mismatch), it reduces peak "
            "control effort and eliminates energy-wasting oscillations. This is a real efficiency gain."
        )

    if "CONTINUITY_ARMOR_EFFECT" in explanations or "CORE_COMPONENT_EFFICIENCY" in explanations:
        return (
            "CONTINUITY ARMOR: The high continuity penalty on the core component creates a 'continuity armor' effect. "
            "By resisting rapid changes in the integral state, it prevents the system from accumulating excessive "
            "integral windup, which would require energy to unwind. This is a genuine 'free lunch' from the "
            "continuity-salience mechanism."
        )

    # Multiple explanations
    top_2 = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    return (
        f"MIXED CAUSES: The {mismatch_ratio:.2f}× mismatch appears to result from multiple factors: "
        f"{top_2[0][0]} (confidence: {top_2[0][1]:.1f}%) and {top_2[1][0]} (confidence: {top_2[1][1]:.1f}%). "
        "The effect is real but the mechanisms are complex."
    )


def main() -> None:
    """Run the diagnostic analysis."""

    print("=" * 80)
    print("ENERGY-MASS MISMATCH DIAGNOSTIC")
    print("=" * 80)
    print()

    # Configuration
    cfg = ExperimentConfig()
    baseline_settings = ContinuitySettings(lambda_core=0.0, lambda_edge=0.0)
    adaptive_settings = ContinuitySettings(lambda_core=40.0, lambda_edge=0.05)

    print("Running baseline simulation (λ_core=0, λ_edge=0)...")
    baseline_traces = simulate_with_traces(baseline_settings, cfg)

    print("Running adaptive simulation (λ_core=40, λ_edge=0.05)...")
    adaptive_traces = simulate_with_traces(adaptive_settings, cfg)

    print("\nComputing component-wise metrics...")
    component_metrics = compute_component_wise_metrics(baseline_traces, adaptive_traces, cfg)

    print("Analyzing energy distribution...")
    energy_dist = analyze_energy_distribution(baseline_traces, adaptive_traces, cfg)

    print("\nGenerating analysis report...")
    report = generate_analysis_report(
        baseline_traces,
        adaptive_traces,
        component_metrics,
        energy_dist,
        cfg,
        baseline_settings,
        adaptive_settings,
    )

    # Create output directory
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save report
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = ARTIFACT_DIR / f"energy_mass_analysis_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Analysis report saved to {report_path}")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_diagnostic_figures(baseline_traces, adaptive_traces, component_metrics, energy_dist, ARTIFACT_DIR)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nAnomaly: {report['anomaly_summary']['description']}")
    print(f"Mismatch Factor: {report['anomaly_summary']['mismatch_factor']}")
    print(f"\nPrimary Explanation: {report['hypotheses']['primary_explanation']}")
    print(f"Confidence: {report['hypotheses']['primary_confidence']:.1f}%")
    print(f"\nVerdict:\n{report['conclusions']['verdict']}")
    print("\n" + "=" * 80)
    print(f"\nFull analysis saved to: {report_path}")
    print(f"Diagnostic plots saved to: {ARTIFACT_DIR / 'diagnostic_plots.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
