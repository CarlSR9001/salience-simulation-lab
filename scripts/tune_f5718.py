"""Parameter sweep for F571-8 using rotmod baryonic data."""

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

DISK_SCALES = [0.6, 0.8, 1.0, 1.2]
BULGE_SCALES = [0.8, 1.0, 1.2]
GAS_SCALES = [1.0, 1.2, 1.4]
HALO_VELOCITIES = [0.0, 30.0, 50.0]  # km/s constant tail added in quadrature


def build_variant(record, config: CSGConfig, *, disk_scale: float, bulge_scale: float, gas_scale: float, halo_velocity: float) -> GalaxyData:
    radii = record.radius_kpc
    v_disk = record.v_disk * np.sqrt(disk_scale)
    v_bulge = record.v_bulge * np.sqrt(bulge_scale)
    v_gas = record.v_gas * gas_scale

    halo = np.full_like(radii, halo_velocity)
    v_bar = np.sqrt(np.clip(v_disk**2 + v_bulge**2 + v_gas**2 + halo**2, 0.0, None))

    metadata = {
        "scenario": "f5718_variant",
        "disk_scale": disk_scale,
        "bulge_scale": bulge_scale,
        "gas_scale": gas_scale,
        "halo_velocity": halo_velocity,
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
    for disk_scale, bulge_scale, gas_scale, halo_velocity in product(
        DISK_SCALES, BULGE_SCALES, GAS_SCALES, HALO_VELOCITIES
    ):
        galaxy = build_variant(
            record,
            config,
            disk_scale=disk_scale,
            bulge_scale=bulge_scale,
            gas_scale=gas_scale,
            halo_velocity=halo_velocity,
        )

        scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)

        residuals = residual_map[TARGET]
        radii = galaxy.radii_kpc
        median_r = np.median(radii)
        inner_mask = radii <= median_r
        outer_mask = radii > median_r
        inner_mean = float(np.mean(residuals[inner_mask]))
        outer_mean = float(np.mean(residuals[outer_mask]))

        results.append(
            {
                "disk": disk_scale,
                "bulge": bulge_scale,
                "gas": gas_scale,
                "halo": halo_velocity,
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
            r["disk"],
            r["bulge"],
            r["gas"],
            r["halo"],
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
                "disk_scale",
                "bulge_scale",
                "gas_scale",
                "halo_v",
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
