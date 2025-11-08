"""Two-zone scaling experiment for SPARC galaxy F571-8."""

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

DISK_SCALES = [0.6, 0.8, 1.0]
DISK_OUTER_SCALES = [0.6, 0.8, 1.0]
GAS_SCALES_INNER = [1.0, 1.3, 1.6]
GAS_SCALES_OUTER = [1.0, 1.4, 1.8]
BREAK_RADII = [4.0, 6.0, 8.0]  # kpc


def build_variant(record, *, disk_inner: float, disk_outer: float, gas_inner: float, gas_outer: float, r_break: float) -> GalaxyData:
    radii = record.radius_kpc
    inner_mask = radii <= r_break
    outer_mask = ~inner_mask

    v_disk = record.v_disk.copy()
    v_disk[inner_mask] *= np.sqrt(disk_inner)
    v_disk[outer_mask] *= np.sqrt(disk_outer)

    v_bulge = record.v_bulge.copy()  # keep as-is (already small in rotmod)

    v_gas = record.v_gas.copy()
    v_gas[inner_mask] *= gas_inner
    v_gas[outer_mask] *= gas_outer

    v_bar_sq = v_disk**2 + v_bulge**2 + v_gas**2
    v_bar = np.sqrt(np.clip(v_bar_sq, 0.0, None))

    metadata = {
        "scenario": "f5718_twozone",
        "disk_inner": disk_inner,
        "disk_outer": disk_outer,
        "gas_inner": gas_inner,
        "gas_outer": gas_outer,
        "r_break": r_break,
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
    for disk_in, disk_out, gas_in, gas_out, r_break in product(
        DISK_SCALES,
        DISK_OUTER_SCALES,
        GAS_SCALES_INNER,
        GAS_SCALES_OUTER,
        BREAK_RADII,
    ):
        galaxy = build_variant(
            record,
            disk_inner=disk_in,
            disk_outer=disk_out,
            gas_inner=gas_in,
            gas_outer=gas_out,
            r_break=r_break,
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
                "disk_in": disk_in,
                "disk_out": disk_out,
                "gas_in": gas_in,
                "gas_out": gas_out,
                "r_break": r_break,
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
            r["disk_in"],
            r["disk_out"],
            r["gas_in"],
            r["gas_out"],
            r["r_break"],
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
                "disk_in",
                "disk_out",
                "gas_in",
                "gas_out",
                "r_break [kpc]",
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
