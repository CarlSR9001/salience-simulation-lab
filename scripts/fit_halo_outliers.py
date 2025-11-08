"""Fit linear mass models (disk/bulge/gas scaling + constant halo) to SPARC outliers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser

TARGETS = ["F568-1", "F568-3", "F571-8", "IC4202"]


def fit_mass_model(record) -> dict[str, float | np.ndarray]:
    v_obs = record.v_obs
    radii = record.radius_kpc

    disk_sq = np.square(record.v_disk)
    bulge_sq = np.square(record.v_bulge)
    gas_sq = np.square(record.v_gas)
    ones = np.ones_like(v_obs)

    mask = np.isfinite(v_obs)
    A = np.column_stack((disk_sq[mask], bulge_sq[mask], gas_sq[mask], ones[mask]))
    y = np.square(v_obs[mask])

    active = [0, 1, 2, 3]
    coeffs = np.zeros(4, dtype=float)

    for _ in range(4):
        A_active = A[:, active]
        x, _, _, _ = np.linalg.lstsq(A_active, y, rcond=None)
        coeffs_tmp = coeffs.copy()
        for idx, col in enumerate(active):
            coeffs_tmp[col] = x[idx]
        negatives = [col for col in active if coeffs_tmp[col] < 0.0]
        if not negatives:
            coeffs = coeffs_tmp
            break
        active = [col for col in active if col not in negatives]
        for col in negatives:
            coeffs[col] = 0.0

    disk_ml, bulge_ml, gas_scale, halo_sq = coeffs
    disk_ml = max(disk_ml, 0.0)
    bulge_ml = max(bulge_ml, 0.0)
    gas_scale = max(gas_scale, 0.0)
    halo_sq = max(halo_sq, 0.0)

    v_bar_sq = disk_ml * disk_sq + bulge_ml * bulge_sq + gas_scale * gas_sq
    v_bar = np.sqrt(np.clip(v_bar_sq, 0.0, None))
    v_halo = np.sqrt(np.clip(halo_sq, 0.0, None))
    v_pred = np.sqrt(v_bar_sq + halo_sq)

    frac_resid = (v_pred - v_obs) / np.maximum(v_obs, 1e-3)
    mean_abs = float(np.mean(np.abs(frac_resid)) * 100.0)
    max_abs = float(np.max(np.abs(frac_resid)) * 100.0)

    return {
        "disk_ml": float(disk_ml),
        "bulge_ml": float(bulge_ml),
        "gas_scale": float(gas_scale),
        "v_halo": float(v_halo),
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "v_bar": v_bar,
        "v_pred": v_pred,
        "v_obs": v_obs,
        "radii": radii,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")

    rows = []
    for name in TARGETS:
        record = parser.load_record(name)
        result = fit_mass_model(record)
        rows.append(
            (
                name,
                record.galaxy_type,
                result["disk_ml"],
                result["bulge_ml"],
                result["gas_scale"],
                result["v_halo"],
                result["mean_abs"],
                result["max_abs"],
            )
        )

        profile_path = root / "artifacts" / f"mass_model_{name}.csv"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            profile_path,
            np.column_stack(
                (
                    result["radii"],
                    result["v_obs"],
                    result["v_bar"],
                    result["v_pred"],
                )
            ),
            delimiter=",",
            header="radius_kpc,v_obs_km_s,v_bar_fit_km_s,v_pred_km_s",
            comments="",
            fmt="%.6f",
        )

    print(
        tabulate(
            rows,
            headers=[
                "Galaxy",
                "Type",
                "Disk M/L",
                "Bulge M/L",
                "Gas scale",
                "v_halo [km/s]",
                "Mean |dv|/v [%]",
                "Max |dv|/v [%]",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
