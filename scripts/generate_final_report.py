"""Generate the final corrected analysis report."""

import json
from datetime import UTC, datetime
from pathlib import Path

# Create the artifacts directory structure
ARTIFACT_DIR = Path("artifacts/energy_anomaly_investigation")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "metadata": {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment": "experiment_e_energy_mass_mismatch_corrected",
        "baseline_lambda_core": 0.0,
        "baseline_lambda_edge": 0.0,
        "adaptive_lambda_core": 40.0,
        "adaptive_lambda_edge": 0.05,
        "analysis_version": "corrected_v2"
    },

    "executive_summary": {
        "finding": "INVALID_COMPARISON_WITH_PARTIAL_GENUINE_EFFECT",
        "headline": "The 6.95× energy-mass mismatch is primarily a measurement artifact from comparing successful control (baseline) to failed control (adaptive λ=40)",
        "key_insight": "The adaptive system never reaches the target (stops at 20% vs 100%), so lower energy usage is from abandoning the task, not efficiency",
        "confidence": "95%",
        "validity_of_original_comparison": "INVALID - comparing apples to oranges"
    },

    "anomaly_details": {
        "reported_m_eff_core": 1.413,
        "reported_energy_ratio": 0.203,
        "reported_mismatch": 6.95,
        "actual_system_m_eff": None,
        "reason_for_null_m_eff": "Adaptive system never reaches 90% threshold within 5-second simulation window"
    },

    "system_performance": {
        "baseline": {
            "reaches_target": True,
            "rise_time_to_90pct": 1.61,
            "final_output": 1.00,
            "final_error_pct": 0.0,
            "total_energy": 4.677,
            "peak_control": 8.02,
            "overshoot_pct": 15.0,
            "energy_before_rise": 1.117,
            "energy_after_rise": 3.560,
            "energy_wasted_on_overshoot_pct": 76.1
        },
        "adaptive": {
            "reaches_target": False,
            "rise_time_to_90pct": None,
            "final_output": 0.20,
            "final_error_pct": 80.0,
            "total_energy": 0.951,
            "peak_control": 7.62,
            "overshoot_pct": 0.0,
            "energy_before_rise": 0.951,
            "energy_after_rise": 0.0,
            "energy_wasted_on_overshoot_pct": 0.0
        }
    },

    "component_analysis": {
        "core_component": {
            "role": "Integral-like accumulator (primary control authority)",
            "baseline_lag": 2.54,
            "adaptive_lag": 3.59,
            "m_eff": 1.413,
            "interpretation_of_m_eff": "Core accumulation is 41% slower due to continuity tax",
            "baseline_energy": 4.877,
            "adaptive_energy": 0.570,
            "energy_ratio": 0.117,
            "interpretation_of_energy": "Core produces 88% less output because it barely accumulates (mass≈35× baseline)",
            "component_mismatch": 12.09,
            "root_cause": "λ_core=40 creates effective gain of 2.86% (1/35), crippling integration",
            "effect": "CATASTROPHIC_OVER_DAMPING - system fails to reach target"
        },
        "edge_component": {
            "role": "Derivative-like rapid response (transient correction)",
            "baseline_lag": 1.36,
            "adaptive_lag": 0.42,
            "m_eff": 0.309,
            "interpretation_of_m_eff": "Edge response is 69% faster (less reactive lag)",
            "baseline_energy": 0.887,
            "adaptive_energy": 0.513,
            "energy_ratio": 0.579,
            "interpretation_of_energy": "Edge uses 42% less energy through mild damping",
            "component_mismatch": 0.533,
            "root_cause": "λ_edge=0.05 creates mild damping without over-suppression",
            "effect": "BENEFICIAL_DAMPING - faster AND more efficient"
        }
    },

    "root_cause_analysis": {
        "primary_cause": {
            "name": "INVALID_METRIC_COMPARISON",
            "confidence": 80,
            "explanation": "Comparing a system that completes its task (baseline: reaches target, corrects overshoot) to a system that fails its task (adaptive: stops at 20% of target). Lower energy in adaptive is from task abandonment, not efficiency.",
            "analogy": "Like comparing fuel efficiency of a car that drives 100 miles vs a car that drives 20 miles then stops"
        },
        "secondary_cause": {
            "name": "EXTREME_CONTINUITY_TAX",
            "confidence": 20,
            "explanation": "λ_core=40 is 10-20× too large, creating a 35× effective mass that reduces integration rate to 2.86% of baseline. This prevents accumulation of sufficient control authority.",
            "calculation": "mass = 1 + λ_core * salience ≈ 1 + 40 * 0.85 = 35"
        },
        "hidden_mechanism": {
            "name": "CONTINUITY_DAMPING_AT_MODERATE_VALUES",
            "status": "OBSCURED_BY_FAILURE",
            "evidence": "Edge component shows genuine efficiency (m_eff=0.31, energy_ratio=0.58) with λ_edge=0.05",
            "hypothesis": "Moderate λ_core (1-5) could reduce overshoot while reaching target, saving the 76% of energy wasted on correction"
        }
    },

    "metric_validity": {
        "m_eff_metric": {
            "definition": "Ratio of component lag (time to 50% of output area) between adaptive and baseline",
            "measures": "Inertia in component's internal state accumulation",
            "does_NOT_measure": "System energy efficiency or ability to reach target",
            "problem": "Component lag ≠ system rise time, especially when outputs have different scales",
            "appropriate_use": "Characterizing component dynamics within a functioning system",
            "inappropriate_use": "Comparing energy efficiency across systems with different goals or success rates"
        },
        "energy_ratio_metric": {
            "definition": "Ratio of total control energy between adaptive and baseline",
            "measures": "Integrated absolute control effort over simulation time",
            "does_NOT_measure": "Efficiency per unit of progress toward target",
            "problem": "Doesn't account for whether target was reached",
            "appropriate_use": "Comparing energy usage when both systems achieve similar final states",
            "inappropriate_use": "Comparing systems where one fails to reach target"
        },
        "energy_mass_mismatch": {
            "definition": "Ratio m_eff / energy_ratio",
            "expected_value": "~1.0 for systems following basic physics (more inertia ⇒ more energy)",
            "observed_value": 6.95,
            "interpretation": "Mismatch indicates metric incompatibility or invalid comparison",
            "corrected_interpretation": "High mismatch is artifact of comparing component-level lag to system-level energy in a failed control scenario"
        }
    },

    "key_insights": {
        "insight_1": {
            "title": "Continuity tax creates inertia, but inertia doesn't imply high energy",
            "explanation": "High m_eff means slow accumulation, which can REDUCE energy by preventing overshoot. The energy-mass metaphor from physics (F=ma ⇒ more mass needs more force) doesn't hold here because control systems aren't trying to accelerate mass, they're trying to accumulate state to a specific target."
        },
        "insight_2": {
            "title": "Energy savings from failure are not efficiency gains",
            "explanation": "The adaptive system uses 80% less energy but achieves only 20% of the target. Normalizing by progress: baseline uses 4.68 energy per 100% progress = 4.68 per unit. Adaptive uses 0.95 energy per 20% progress = 4.75 per unit. Actually LESS efficient when accounting for partial completion."
        },
        "insight_3": {
            "title": "Component-wise analysis reveals opposing trends",
            "explanation": "Core shows catastrophic over-damping (12× mismatch), edge shows beneficial damping (0.5× mismatch). System-level metrics mask this critical distinction."
        },
        "insight_4": {
            "title": "Baseline wastes 76% of energy on overshoot correction",
            "explanation": "A properly tuned continuity tax could eliminate this waste by acting as an 'integral damper', but λ_core=40 overshoots in the opposite direction, creating under-response instead of optimal response."
        }
    },

    "recommendations": {
        "immediate": {
            "action": "RERUN_WITH_LAMBDA_SWEEP",
            "parameters": {
                "lambda_core_values": [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 40],
                "lambda_edge": 0.05,
                "horizon": 10.0
            },
            "metrics_to_track": [
                "final_output (must be ≥0.95 to be valid)",
                "rise_time_to_90pct",
                "total_energy",
                "energy_after_rise (overshoot correction)",
                "peak_overshoot",
                "m_eff / energy_ratio mismatch"
            ],
            "expected_finding": "Goldilocks zone at λ_core≈2-5 where system reaches target with 30-60% energy savings"
        },
        "analysis": {
            "action": "NORMALIZE_ENERGY_BY_PROGRESS",
            "method": "energy_per_unit_progress = total_energy / (final_output / target)",
            "rationale": "Fair comparison requires accounting for how close each system gets to target"
        },
        "metric_design": {
            "action": "DEFINE_VALID_COMPARISON_CRITERIA",
            "criteria": [
                "Both systems must reach ≥95% of target",
                "Compare rise times only if both successfully rise",
                "Separate pre-rise energy (approach) from post-rise energy (correction)",
                "Report m_eff and energy_ratio per component with context"
            ]
        }
    },

    "hypothesis_for_goldilocks_zone": {
        "claim": "There exists λ_core ∈ [2, 5] where continuity tax creates genuine efficiency",
        "mechanism": "Moderate continuity penalty damps integral accumulation just enough to prevent overshoot without preventing target achievement",
        "predicted_metrics": {
            "final_output": ">0.95",
            "rise_time_increase": "20-40%",
            "total_energy_decrease": "30-60%",
            "overshoot_energy_savings": ">70%",
            "m_eff_energy_ratio_mismatch": "1.0-2.0"
        },
        "test": "Run sweep and find λ_core that minimizes (total_energy) subject to (final_output ≥ 0.95)"
    },

    "final_verdict": {
        "question_1_is_adaptive_more_efficient": {
            "answer": "NOT AT λ_core=40",
            "elaboration": "The adaptive system fails to reach the target, so it's not 'efficient' but 'failed'. However, the edge component (λ_edge=0.05) shows genuine efficiency, suggesting moderate λ_core could work."
        },
        "question_2_rise_time_calculation_missing_something": {
            "answer": "YES - COMPONENT LAG ≠ SYSTEM RISE TIME",
            "elaboration": "Component lag measures internal state accumulation (time to 50% of output area). System rise time measures target achievement (time to 90% of target). These are different, especially when component outputs have different scales."
        },
        "question_3_energy_needs_component_separation": {
            "answer": "YES - CRITICAL FOR UNDERSTANDING",
            "elaboration": "Component-wise analysis reveals core (12× mismatch) vs edge (0.5× mismatch), opposing effects that cancel in aggregate metrics."
        },
        "question_4_effective_mass_metaphor_appropriate": {
            "answer": "PARTIALLY - MISLEADING FOR ENERGY",
            "elaboration": "The metaphor correctly captures 'inertia' in state dynamics but incorrectly implies high inertia ⇒ high energy. In control systems, high inertia can reduce energy by preventing overshoot."
        },
        "question_5_is_mismatch_real_or_artifact": {
            "answer": "80% ARTIFACT, 20% REAL EFFECT OBSCURED",
            "elaboration": "Primarily artifact from invalid comparison (failed vs successful control). But edge component shows genuine efficiency, and moderate λ_core likely would too."
        }
    },

    "conclusion": "The 6.95× energy-mass mismatch is NOT a 'free lunch' from continuity armor at λ_core=40. It's an invalid comparison between a system that reaches its target and one that doesn't. However, this investigation reveals a promising research direction: moderate continuity penalties could create genuine efficiency by eliminating the 76% of energy wasted on overshoot correction in the baseline. Recommended next step: λ_core sweep from 0 to 10 to find the optimal damping that maximizes energy efficiency while maintaining target achievement."
}

# Save the report
timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
output_path = ARTIFACT_DIR / f"energy_mass_analysis_{timestamp}.json"
output_path.write_text(json.dumps(report, indent=2))
print(f"Final corrected analysis saved to: {output_path}")
print(f"\nKey finding: {report['executive_summary']['headline']}")
print(f"\nConclusion: {report['conclusion']}")
