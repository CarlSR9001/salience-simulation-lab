"""Experiment E: Energy Accounting / Proto-Landauer Check."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

ARTIFACT_DIR = Path("artifacts/mass_sweep")


def _latest_matching(prefix: str) -> Optional[Path]:
    candidates = sorted(ARTIFACT_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@dataclass
class MassPoint:
    lambda_c: float
    m_eff: float
    control_energy: float


def load_mass_sweep(path: Path) -> List[MassPoint]:
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return [
        MassPoint(
            lambda_c=float(record["lambda_c"]),
            m_eff=float(record["m_eff"]),
            control_energy=float(record["control_energy"]),
        )
        for record in records
    ]


@dataclass
class AdaptivePoint:
    lambda_core: float
    lambda_edge: float
    m_eff_core: float
    m_eff_edge: float
    control_energy: float


def load_adaptive(path: Path) -> List[AdaptivePoint]:
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return [
        AdaptivePoint(
            lambda_core=float(record["lambda_core"]),
            lambda_edge=float(record["lambda_edge"]),
            m_eff_core=float(record["m_eff_core"]),
            m_eff_edge=float(record["m_eff_edge"]),
            control_energy=float(record.get("control_energy", math.nan)),
        )
        for record in records
    ]


def compute_energy_ratios(mass_points: List[MassPoint]) -> List[Dict[str, float]]:
    mass_points = sorted(mass_points, key=lambda p: p.lambda_c)
    baseline = mass_points[0]
    results: List[Dict[str, float]] = []
    for point in mass_points[1:]:
        energy_ratio = point.control_energy / baseline.control_energy if baseline.control_energy else math.nan
        results.append(
            {
                "lambda_c": point.lambda_c,
                "m_eff": point.m_eff,
                "baseline_control_energy": baseline.control_energy,
                "taxed_control_energy": point.control_energy,
                "energy_ratio": energy_ratio,
            }
        )
    return results


def compute_adaptive_energy(adaptive_points: List[AdaptivePoint]) -> List[Dict[str, float]]:
    if len(adaptive_points) < 2:
        return []
    baseline = adaptive_points[0]
    taxed = adaptive_points[-1]
    energy_ratio = taxed.control_energy / baseline.control_energy if baseline.control_energy else math.nan
    return [
        {
            "lambda_core_baseline": baseline.lambda_core,
            "lambda_edge_baseline": baseline.lambda_edge,
            "lambda_core_taxed": taxed.lambda_core,
            "lambda_edge_taxed": taxed.lambda_edge,
            "m_eff_core": taxed.m_eff_core,
            "m_eff_edge": taxed.m_eff_edge,
            "baseline_control_energy": baseline.control_energy,
            "taxed_control_energy": taxed.control_energy,
            "energy_ratio": energy_ratio,
        }
    ]


def write_artifact(payload: Dict[str, Union[List[Dict[str, float]], Dict[str, str]]]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = ARTIFACT_DIR / f"mass_energy_{timestamp}.json"
    for section in payload.values():
        if isinstance(section, list):
            for entry in section:
                entry["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                entry["experiment_name"] = "experiment_e_energy_accounting"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def main() -> None:
    mass_path = _latest_matching("mass_sweep_")
    adaptive_path = _latest_matching("adaptive_mass_")
    if mass_path is None or adaptive_path is None:
        raise FileNotFoundError("Required mass sweep or adaptive mass artifacts not found")

    mass_points = load_mass_sweep(mass_path)
    adaptive_points = load_adaptive(adaptive_path)

    mass_energy = compute_energy_ratios(mass_points)
    adaptive_energy = compute_adaptive_energy(adaptive_points)

    payload: Dict[str, Union[List[Dict[str, float]], Dict[str, str]]] = {
        "mass_sweep": mass_energy,
        "adaptive": adaptive_energy,
        "sources": {
            "mass_sweep_artifact": str(mass_path),
            "adaptive_artifact": str(adaptive_path),
        },
    }

    path = write_artifact(payload)
    print(f"Energy accounting written to {path}")


if __name__ == "__main__":
    main()
