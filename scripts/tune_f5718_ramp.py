"""Radial ramp scaling for SPARC galaxy F571-8."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.galaxy import GalaxyData
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics

TARGET = "F571-8"

DISK_BASE = [0.5, 0.6, 0.8]
DISK_SLOPE = [-0.3, -0.1, 0.0, 0.2]
GAS_BASE = [1.0, 1.2]
GAS_SLOPE = [0.0, 0.3, 0.6, 0.9]


def build_variant(record, *, disk_base: float, disk_slope: float, gas_base: float, gas_slope: float) -> GalaxyData:
    radii = record.radius_kpc
    r_norm = radii / np.max(radii)

    disk_scale = np.clip(disk_base + disk_slope * r_norm, 0.1, 2.5)
    gas_scale = np.clip(gas_base + gas_slope * r_norm, 0.1, 3.0)

    v_disk = record.v_disk * np.sqrt(disk_scale)
    v_bulge = record.v_bulge
    v_gas = record.v_gas * gas_scale

    v_bar_sq = v_disk**2 + v_bulge**2 + v_gas**2
    v_bar = np.sqrt(np.clip(v_bar_sq, 0.0, None))

    metadata = {
        "scenario": "f5718_ramp",
        "disk_base": disk_base,
        "disk_slope": disk_slope,
        "gas_base": gas_base,
        "gas_slope": gas_slope,
    }

    return GalaxyData(
        name=record.galaxy,
        galaxy_type=record.galaxy_type,
        radii_kpc=radii,
        v_obs=record.v_obs,
        v_bar=v_bar,
        sigma_v=np.maximum(record.e_v_obs, 0.5),
        gas_fraction=record.gas_fraction,
        age_gyr=10.0,
        has_coherent_rotation=True,
        metadata=metadata,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    record = parser.load_record(TARGET)

    results = []
    for disk_base, disk_slope, gas_base, gas_slope in product(
        DISK_BASE,
        DISK_SLOPE,
        GAS_BASE,
        GAS_SLOPE,
    ):
        galaxy = build_variant(
            record,
            disk_base=disk_base,
            disk_slope=disk_slope,
            gas_base=gas_base,
            gas_slope=gas_slope,
        )

        scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)

        residuals = residual_map[TARGET]
        radii = galaxy.radii_kpc
        median_r = np.median(radii)
        inner_mean = float(np.mean(residuals[radii <= median_r]))
        outer_mean = float(np.mean(residuals[radii > median_r]))

        results.append(
            {
                "disk_base": disk_base,
                "disk_slope": disk_slope,
                "gas_base": gas_base,
                "gas_slope": gas_slope,
                "kappa": best_kappa,
                "mean_abs": metrics.mean_percent_error,
                "max_abs": metrics.max_percent_error,
                "inner_mean": inner_mean,
                "outer_mean": outer_mean,
            }
        )

    results.sort(key=lambda r: r["mean_abs"])
    top_rows = [
        (
            r["disk_base"],
            r["disk_slope"],
            r["gas_base"],
            r["gas_slope"],
            r["kappa"],
            r["mean_abs"],
            r["max_abs"],
            r["inner_mean"],
            r["outer_mean"],
        )
        for r in results[:10]
    ]

    print(
        tabulate(
            top_rows,
            headers=[
                "disk_base",
                "disk_slope",
                "gas_base",
                "gas_slope",
                "kappa_c",
                "Mean |dv|/v [%]",
                "Max |dv|/v [%]",
                "Inner mean",
                "Outer mean",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
