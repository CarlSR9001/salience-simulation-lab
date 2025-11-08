"""Empirical validation pipeline for SAL using real SPARC data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from csg_v4.analysis import CSGAnalysis
from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.reporting import GalaxyMetrics


def _serialize_metrics(metrics: Iterable[GalaxyMetrics]) -> list[dict[str, float | int | str]]:
    return [asdict(metric) for metric in metrics]


def _build_summary(
    metrics: Sequence[GalaxyMetrics],
    best_kappa: float,
    best_chisq: float,
    bootstrap_samples: np.ndarray,
    weighted_best_kappa: float | None,
    weighted_chisq: np.ndarray | None,
) -> dict[str, object]:
    mean_percent = float(np.mean([m.mean_percent_error for m in metrics]))
    max_percent = float(np.max([m.max_percent_error for m in metrics]))
    rms = float(np.mean([m.rms_error for m in metrics]))

    summary: dict[str, object] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "num_galaxies": len(metrics),
        "best_kappa": float(best_kappa),
        "best_chisq": float(best_chisq),
        "bootstrap_mean": float(np.mean(bootstrap_samples)),
        "bootstrap_std": float(np.std(bootstrap_samples)),
        "mean_percent_error": mean_percent,
        "max_percent_error": max_percent,
        "mean_rms_error_kms": rms,
    }

    if weighted_best_kappa is not None:
        summary["radius_weighted_best_kappa"] = float(weighted_best_kappa)
    if weighted_chisq is not None:
        summary["radius_weighted_min_chisq"] = float(np.min(weighted_chisq))

    return summary


def run_empirical_validation(
    galaxies: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    top_n: int | None = 25,
) -> Path:
    """Run the empirical validation pipeline and return the artifact directory."""

    config = CSGConfig()
    parser = SPARCParser(Path("data/sparc/table1.dat"), Path("data/sparc/table2.dat"))

    available = parser.list_galaxies()
    if galaxies:
        selection = list(galaxies)
    else:
        selection = available[:top_n] if top_n is not None else available

    if not selection:
        raise ValueError("No galaxies selected for empirical validation")

    galaxy_map = parser.load_galaxies(selection, config)

    if output_dir is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path("artifacts") / "empirical_validation" / timestamp
    else:
        artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    analysis = CSGAnalysis(config)
    artifacts = analysis.run(galaxies=galaxy_map.values(), output_dir=artifact_dir)

    metrics = list(artifacts.metrics.values())
    metrics.sort(key=lambda metric: metric.mean_percent_error)

    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_serialize_metrics(metrics), indent=2))

    summary = _build_summary(
        metrics,
        artifacts.scan_result.best_kappa,
        artifacts.scan_result.best_chisq,
        artifacts.bootstrap_kappas,
        artifacts.weighted_best_kappa,
        artifacts.weighted_chisq,
    )

    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    readme_path = artifact_dir / "README.txt"
    readme_path.write_text(
        "Empirical validation artifacts generated from SPARC rotation curves.\n"
        "Files:\n"
        "  - summary.json: aggregate fit statistics across galaxies.\n"
        "  - metrics.json: per-galaxy residual metrics sorted by mean percent error.\n"
        "  - kappa_scan.png / rotation_curve_*.png: visual diagnostics from the run.\n"
    )

    return artifact_dir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run empirical validation using SPARC data")
    parser.add_argument(
        "--galaxy",
        dest="galaxies",
        action="append",
        help="Specific galaxy name to include (can be repeated).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Use the first N SPARC galaxies when --galaxy is not provided (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom directory for artifacts (defaults to artifacts/empirical_validation/<timestamp>).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_empirical_validation(galaxies=args.galaxies, output_dir=args.output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
