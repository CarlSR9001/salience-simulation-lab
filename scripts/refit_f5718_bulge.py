"""Refit F571-8 disk/bulge profiles from SPARC surface-density data."""

from __future__ import annotations

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

G = 4.302e-6  # kpc (km/s)^2 / Msun, gravitational constant
ML_DISK = 0.6  # default SPARC value


def load_surface_density(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    radii = data[:, 0]
    sb_disk = data[:, 1]
    sb_bulge = data[:, 2]
    return radii, sb_disk, sb_bulge


def compute_velocity_from_sb(radii: np.ndarray, sb: np.ndarray, ml: float) -> np.ndarray:
    # Approximate enclosed mass via cumulative sum of surface density * annulus area
    radii, sb = np.asarray(radii), np.asarray(sb)
    mass = np.zeros_like(radii)
    for i in range(1, radii.size):
        r_outer = radii[i]
        r_inner = radii[i - 1]
        area = np.pi * (r_outer**2 - r_inner**2)
        surface_density = 0.5 * (sb[i] + sb[i - 1]) * ml  # average M/L weighting
        mass[i] = mass[i - 1] + surface_density * area
    mass = np.maximum(mass, 0.0)
    velocity = np.sqrt(np.clip(G * mass / np.maximum(radii, 1e-3), 0.0, None))
    return velocity


def build_galaxy(record, config: CSGConfig, refit_disk: np.ndarray) -> GalaxyData:
    v_gas = record.v_gas
    v_bulge = record.v_bulge
    v_bar_sq = refit_disk**2 + v_bulge**2 + v_gas**2
    v_bar = np.sqrt(np.clip(v_bar_sq, 0.0, None))
    metadata = {"scenario": "f5718_refit_disk"}
    return GalaxyData(
        name=record.galaxy,
        galaxy_type=record.galaxy_type,
        radii_kpc=record.radius_kpc,
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
    record = parser.load_record(TARGET)

    sfb_path = data_dir / "extended" / "extracted" / "bulge" / f"{TARGET}.dens"
    radii_sb, sb_disk, sb_bulge = load_surface_density(sfb_path)

    vel_disk_refit = compute_velocity_from_sb(radii_sb, sb_disk, ML_DISK)

    # Resample to match rotation radii using interpolation
    vel_disk_interp = np.interp(record.radius_kpc, radii_sb, vel_disk_refit, left=vel_disk_refit[0], right=vel_disk_refit[-1])

    config = CSGConfig()
    model = CSGV4Model(config)

    galaxy_refit = build_galaxy(record, config, vel_disk_interp)
    scan, residual_map = scan_kappa([galaxy_refit], model=model, store_profiles=True)
    best_kappa = scan.best_kappa
    outputs = model.predict_velocity(galaxy_refit, best_kappa)
    metrics = compute_error_metrics(galaxy_refit, outputs)

    residuals = residual_map[TARGET]
    radii = galaxy_refit.radii_kpc
    median_r = np.median(radii)
    inner_mean = float(np.mean(residuals[radii <= median_r]))
    outer_mean = float(np.mean(residuals[radii > median_r]))

    print(
        tabulate(
            [
                (
                    TARGET,
                    galaxy_refit.galaxy_type,
                    best_kappa,
                    metrics.mean_percent_error,
                    metrics.max_percent_error,
                    inner_mean,
                    outer_mean,
                )
            ],
            headers=["Galaxy", "Type", "kappa_c", "Mean |dv|/v [%]", "Max |dv|/v [%]", "Inner mean", "Outer mean"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
