"""Experiment W: Authority re-entry ramp probe.

This bridges the continuity mass-stress campaign with the later Lorenz gating
and predictive re-entry tactics documented in the execution plan. The script
replays the hostile Lorenz forcing scenario from Experiment P but compares
three remediation strategies:

1. **Ungated chaos** – baseline showing loss of authority.
2. **Ramped gate** – authority must stay above a threshold for a sustained
   window before Lorenz forcing ramps back on.
3. **Predictive re-entry** – combines the ramped gate with a Poincaré-style
   predictive assist that stages the re-introduction of chaos when authority
   momentum is positive.

Each scenario reuses the continuity PID and salience bookkeeping from
``scripts.continuity_mass_sim`` so results can be compared directly with the
earlier stress logs. Metrics are written to ``artifacts/mass_sweep`` with a
distinct prefix and summarised on stdout.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import numpy as np

from scripts.continuity_mass_sim import (
    ControllerConfig,
    PlantConfig,
    run_stress_sweep,
    write_artifact,
)

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None  # type: ignore[assignment]


LAMBDAS_DEFAULT = [0.0, 5.0, 10.0]
SALIENCE_FLOOR = 0.6


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    overrides: Dict[str, object]


SCENARIOS: List[Scenario] = [
    Scenario(
        name="ungated_lorenz",
        description="Lorenz forcing at gain 0.18 without any gating",
        overrides={
            "lorenz_gain": 0.18,
            "lorenz_gate_enabled": False,
            "lorenz_hot_tag_enabled": False,
            "adaptive_breathing": False,
            "horizon": 8.0,
            "salience_channels": 4,
        },
    ),
    Scenario(
        name="ramped_gate",
        description="Lorenz forcing ramped back after sustained authority",
        overrides={
            "lorenz_gain": 0.18,
            "lorenz_gate_enabled": True,
            "lorenz_gate_threshold": 1.05,
            "lorenz_gate_release_steps": 20,
            "lorenz_ramp_steps": 60,
            "adaptive_breathing": True,
            "breath_period": 6.0,
            "breath_in_fraction": 0.4,
            "breath_in_scale": 0.08,
            "breath_out_scale": 0.5,
            "horizon": 10.0,
            "salience_channels": 4,
        },
    ),
    Scenario(
        name="predictive_reentry",
        description=(
            "Ramped gate + Poincaré assist and salience recovery before chaos"
        ),
        overrides={
            "lorenz_gain": 0.18,
            "lorenz_gate_enabled": True,
            "lorenz_gate_threshold": 1.08,
            "lorenz_gate_release_steps": 18,
            "lorenz_ramp_steps": 45,
            "adaptive_breathing": True,
            "breath_period": 6.0,
            "breath_in_fraction": 0.5,
            "breath_in_scale": 0.06,
            "breath_out_scale": 0.4,
            "salience_breathing": True,
            "salience_breath_floor": 0.6,
            "salience_breath_gain": 2.5,
            "salience_breath_recovery_scale": 0.2,
            "assist_poincare_enabled": True,
            "assist_poincare_axis": 0,
            "assist_poincare_threshold": 18.0,
            "assist_poincare_direction": 1,
            "assist_poincare_window": 6,
            "assist_amplitude": 0.18,
            "assist_pulse_width": 3,
            "assist_cooldown_steps": 25,
            "hope_spot_lower": 0.72,
            "hope_spot_upper": 0.96,
            "lorenz_hot_tag_enabled": True,
            "lorenz_hot_tag_window": 25,
            "lorenz_hot_tag_slope_threshold": 0.18,
            "lorenz_hot_tag_pulse_steps": 35,
            "lorenz_hot_tag_cooldown_steps": 45,
            "lorenz_hot_tag_control_boost": 1.15,
            "horizon": 12.0,
            "salience_channels": 4,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment W: authority re-entry ramp probe"
    )
    parser.add_argument(
        "--lambda-values",
        type=str,
        default=",".join(str(v) for v in LAMBDAS_DEFAULT),
        help="Comma-separated λ_c values",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Base random seed for reproducibility",
    )
    return parser.parse_args()


def parse_lambda_values(payload: str) -> List[float]:
    values: List[float] = []
    for part in payload.split(","):
        label = part.strip()
        if not label:
            continue
        values.append(float(label))
    if not values:
        raise ValueError("Expected at least one λ_c value")
    return values


def summarise(entries: List[Dict[str, float]], scenario: Scenario) -> Dict[str, float]:
    authority = [float(e.get("authority_ok", 0.0)) for e in entries]
    energy = [e.get("energy_ratio") for e in entries]
    m_eff = [e.get("m_eff") for e in entries]
    lorenz_mean = [e.get("lorenz_scale_mean") for e in entries]
    authority_count = int(np.sum(authority))
    def safe_mean(values: Iterable[float | None]) -> float:
        usable = [v for v in values if v is not None and not math.isnan(v)]
        return float(np.mean(usable)) if usable else float("nan")

    return {
        "scenario": scenario.name,
        "authority_successes": authority_count,
        "mean_energy_ratio": safe_mean(energy),
        "mean_m_eff": safe_mean(m_eff),
        "mean_lorenz_scale": safe_mean(lorenz_mean),
    }


def main() -> None:
    args = parse_args()
    lambdas = parse_lambda_values(args.lambda_values)

    base_plant = PlantConfig(dt=0.01, tau=0.1, noise_sigma=0.0)
    controller_template = ControllerConfig(salience_floor=SALIENCE_FLOOR, salience_channels=4)

    all_entries: List[Dict[str, float]] = []
    summaries: List[Dict[str, float]] = []

    print("=== Experiment W: Authority Re-Entry Ramp Probe ===")
    print(f"λ sweep: {', '.join(f'{lam:.1f}' for lam in lambdas)}")
    print()

    for scenario in SCENARIOS:
        plant_cfg = replace(base_plant, **scenario.overrides)
        print(f"Scenario: {scenario.name}")
        print(f"  {scenario.description}")
        results = run_stress_sweep(
            lambdas,
            plant_cfg,
            seed=args.seed,
            salience_floor=SALIENCE_FLOOR,
            controller_template=controller_template,
        )
        for entry in results:
            entry["scenario"] = scenario.name
            entry["scenario_description"] = scenario.description
        summaries.append(summarise(results, scenario))
        all_entries.extend(results)

        if tabulate:
            rows = [
                (
                    entry["lambda_c"],
                    entry.get("authority_ok"),
                    entry.get("control_authority_ratio"),
                    entry.get("m_eff"),
                    entry.get("energy_ratio"),
                    entry.get("lorenz_scale_mean"),
                    entry.get("mean_salience"),
                )
                for entry in results
            ]
            headers = [
                "lambda_c",
                "authority_ok",
                "authority_ratio",
                "m_eff",
                "energy_ratio",
                "lorenz_scale_mean",
                "mean_salience",
            ]
            print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f"))
        else:  # pragma: no cover - tabulate not installed
            for entry in results:
                print(entry)
        print()

    # Attach scenario-level summaries to the artifact payload
    for summary in summaries:
        summary_entry = {
            "scenario": summary["scenario"],
            "authority_successes": summary["authority_successes"],
            "mean_energy_ratio": summary["mean_energy_ratio"],
            "mean_m_eff": summary["mean_m_eff"],
            "mean_lorenz_scale": summary["mean_lorenz_scale"],
            "entry_type": "scenario_summary",
        }
        all_entries.append(summary_entry)

    artifact = write_artifact(
        all_entries,
        experiment_name="experiment_w_authority_reentry",
        prefix="authority_reentry",
    )

    print("Scenario summaries:")
    if tabulate:
        rows = [
            (
                summary["scenario"],
                summary["authority_successes"],
                summary["mean_energy_ratio"],
                summary["mean_m_eff"],
                summary["mean_lorenz_scale"],
            )
            for summary in summaries
        ]
        headers = [
            "scenario",
            "authority_successes",
            "mean_energy_ratio",
            "mean_m_eff",
            "mean_lorenz_scale",
        ]
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f"))
    else:  # pragma: no cover
        for summary in summaries:
            print(summary)

    print(f"Results written to {artifact}")


if __name__ == "__main__":
    main()

