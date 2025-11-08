"""Calibrate sigma coupling for rotmod-enhanced SPARC galaxies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa

TARGETS = ["F563-1", "F568-1", "F568-3", "F571-8", "IC4202"]

MSUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19


def compute_density_profiles(galaxy, config: CSGConfig) -> Dict[str, np.ndarray]:
    r = galaxy.radii_kpc
    v_obs = galaxy.v_obs
    v_bar = galaxy.v_bar

    mass_obs = (np.square(v_obs) * r) / config.G
    mass_bar = (np.square(v_bar) * r) / config.G
    delta_mass = mass_obs - mass_bar

    volume = (4.0 / 3.0) * np.pi * np.power(r, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_mass = np.where(volume > 0.0, delta_mass / volume, 0.0)

    rho_mass = np.nan_to_num(rho_mass, nan=0.0, posinf=0.0, neginf=0.0)

    rho_mass_si = rho_mass * MSUN_KG / (KPC_M**3)
    c_ms = config.c * 1_000.0
    rho_energy = rho_mass_si * c_ms**2

    return {
        "delta_mass": delta_mass,
        "rho_mass_msun_kpc3": rho_mass,
        "rho_energy_si": rho_energy,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    rows = []

    for name in TARGETS:
        record = parser.load_record(name)
        galaxy = parser.to_galaxy_data(record, config)
        density = compute_density_profiles(galaxy, config)

        scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        s_prime = outputs.s_prime

        rho_energy = density["rho_energy_si"]
        mask = s_prime > 1.0e-8
        if not np.any(mask):
            continue

        s_sel = s_prime[mask]
        rho_sel = rho_energy[mask]

        alpha = float(np.dot(s_sel, rho_sel) / np.dot(s_sel, s_sel))
        sigma_samples = np.where(s_sel > 0.0, rho_sel / s_sel, 0.0)

        s_ref = float(np.median(s_sel)) if np.any(s_sel) else 1.0
        info_density = s_sel / (s_ref + 1.0e-12)
        sigma_info_samples = np.where(info_density > 0.0, rho_sel / info_density, 0.0)

        rho_mass_outer = float(density["rho_mass_msun_kpc3"][-1])
        rho_energy_outer = float(rho_energy[-1])

        rows.append(
            (
                name,
                best_kappa,
                alpha,
                float(np.median(sigma_samples)),
                float(np.mean(sigma_samples)),
                float(np.median(sigma_info_samples)),
                float(np.mean(sigma_info_samples)),
                rho_mass_outer,
                rho_energy_outer,
            )
        )

    headers = [
        "Galaxy",
        "Best κ_c",
        "α (J/m³ per S′)",
        "σ̃ median",
        "σ̃ mean",
        "σ_info median [J/m³ per unit]",
        "σ_info mean [J/m³ per unit]",
        "ρΔM outer [M☉/kpc³]",
        "ρΔE outer [J/m³]",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".6g"))


if __name__ == "__main__":
    main()
