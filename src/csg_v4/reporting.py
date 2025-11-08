"""Reporting utilities for the CSG V4 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tabulate import tabulate

from .galaxy import GalaxyData
from .model import CSGV4Model, ModelOutputs


@dataclass(frozen=True)
class GalaxyMetrics:
    name: str
    rms_error: float
    mean_percent_error: float
    max_percent_error: float
    n_points: int


def compute_error_metrics(galaxy: GalaxyData, outputs: ModelOutputs) -> GalaxyMetrics:
    residuals = CSGV4Model.residuals(galaxy, outputs)
    abs_resid = np.abs(residuals)

    rms_error = float(np.sqrt(np.mean(np.square(outputs.v_pred - galaxy.v_obs))))
    mean_percent_error = float(np.mean(abs_resid) * 100.0)
    max_percent_error = float(np.max(abs_resid) * 100.0)

    return GalaxyMetrics(
        name=galaxy.name,
        rms_error=rms_error,
        mean_percent_error=mean_percent_error,
        max_percent_error=max_percent_error,
        n_points=galaxy.n_radii,
    )


def summarize_galaxy_metrics(metrics: Iterable[GalaxyMetrics]) -> str:
    rows: list[tuple] = []
    metrics_list = list(metrics)
    for metric in metrics_list:
        rows.append(
            (
                metric.name,
                metric.n_points,
                f"{metric.rms_error:.2f}",
                f"{metric.mean_percent_error:.2f}",
                f"{metric.max_percent_error:.2f}",
            )
        )
    mean_percent = np.mean([m.mean_percent_error for m in metrics_list])
    table = tabulate(
        rows,
        headers=[
            "Galaxy",
            "N",
            "RMS Error [km/s]",
            "Mean |dv|/v [%]",
            "Max |dv|/v [%]",
        ],
        tablefmt="github",
    )
    overall = f"Overall mean percent error: {mean_percent:.2f}%"
    return table + "\n" + overall
