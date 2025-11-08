"""Sweep outer-disk adjustment parameters for SPARC spiral outliers."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.galaxy import GalaxyData
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics

TARGETS = ["F568-1", "F568-3", "F571-8", "IC4202"]

DISK_ML_MAP = {
    "hsb_spiral": 0.8,
    "spiral": 0.75,
    "grand_design": 0.75,
}
BULGE_ML_MAP = {
    "hsb_spiral": 0.9,
    "spiral": 0.9,
    "grand_design": 0.9,
}
DEFAULT_DISK_ML = 0.6
DEFAULT_BULGE_ML = 0.8

HALO_VELOCITIES = [0.0, 70.0, 100.0]  # km/s
HALO_THRESHOLDS = [6.0, 9.0]  # kpc
GAS_SCALES = [1.0, 1.4, 1.8]
GAS_THRESHOLDS = [6.0, 8.0]
BULGE_SCALES = [1.0, 1.5]


def build_v_components(record) -> Dict[str, np.ndarray]:
    disk_ml = DISK_ML_MAP.get(record.galaxy_type, DEFAULT_DISK_ML)
    bulge_ml = BULGE_ML_MAP.get(record.galaxy_type, DEFAULT_BULGE_ML)
    v_disk = record.v_disk * np.sqrt(disk_ml)
    v_bulge = record.v_bulge * np.sqrt(bulge_ml)
    v_gas = np.abs(record.v_gas)
    sigma = np.maximum(record.e_v_obs, 0.5)
    return {
        "disk": v_disk,
        "bulge": v_bulge,
        "gas": v_gas,
        "sigma": sigma,
    }


def construct_variant(record, base_galaxy: GalaxyData, components: Dict[str, np.ndarray], *,
                      halo_velocity: float,
                      halo_threshold: float,
                      gas_scale: float,
                      gas_threshold: float,
                      bulge_scale: float) -> GalaxyData:
    radii = record.radius_kpc
    v_disk = components["disk"]
    v_bulge = components["bulge"] * bulge_scale

    v_gas = components["gas"].copy()
    mask_gas = radii >= gas_threshold
    v_gas[mask_gas] *= gas_scale

    v_components = v_disk**2 + v_bulge**2 + v_gas**2

    if halo_velocity > 0.0:
        halo = np.zeros_like(radii)
        halo[radii >= halo_threshold] = halo_velocity
        v_components += halo**2

    v_bar = np.sqrt(v_components)
    return GalaxyData(
        name=base_galaxy.name,
        galaxy_type=base_galaxy.galaxy_type,
        radii_kpc=base_galaxy.radii_kpc,
        v_obs=base_galaxy.v_obs,
        v_bar=v_bar,
        sigma_v=components["sigma"],
        gas_fraction=base_galaxy.gas_fraction,
        age_gyr=base_galaxy.age_gyr,
        has_coherent_rotation=base_galaxy.has_coherent_rotation,
        metadata={
            "halo_v": halo_velocity,
            "halo_r": halo_threshold,
            "gas_scale": gas_scale,
            "gas_r": gas_threshold,
            "bulge_scale": bulge_scale,
        },
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    rows = []
    for name in TARGETS:
        record = parser.load_record(name)
        components = build_v_components(record)
        base_galaxy = parser.to_galaxy_data(
            record,
            config,
            stellar_ml=DEFAULT_DISK_ML,
            stellar_ml_map=DISK_ML_MAP,
            bulge_ml=DEFAULT_BULGE_ML,
            bulge_ml_map=BULGE_ML_MAP,
        )

        total_configs = (
            len(HALO_VELOCITIES)
            * len(HALO_THRESHOLDS)
            * len(GAS_SCALES)
            * len(GAS_THRESHOLDS)
            * len(BULGE_SCALES)
        )
        print(f"Sweeping {total_configs} configurations for {name}...")

        for halo_v, halo_r, gas_scale, gas_r, bulge_scale in product(
            HALO_VELOCITIES,
            HALO_THRESHOLDS,
            GAS_SCALES,
            GAS_THRESHOLDS,
            BULGE_SCALES,
        ):
            galaxy_variant = construct_variant(
                record,
                base_galaxy,
                components,
                halo_velocity=halo_v,
                halo_threshold=halo_r,
                gas_scale=gas_scale,
                gas_threshold=gas_r,
                bulge_scale=bulge_scale,
            )
            scan, _ = scan_kappa([galaxy_variant], model=model, store_profiles=False)
            best_kappa = scan.best_kappa
            outputs = model.predict_velocity(galaxy_variant, best_kappa)
            metrics = compute_error_metrics(galaxy_variant, outputs)
            rows.append(
                {
                    "name": name,
                    "halo_v": halo_v,
                    "halo_r": halo_r,
                    "gas_scale": gas_scale,
                    "gas_r": gas_r,
                    "bulge_scale": bulge_scale,
                    "kappa": best_kappa,
                    "mean_abs": metrics.mean_percent_error,
                    "max_abs": metrics.max_percent_error,
                }
            )

            done = len([r for r in rows if r["name"] == name])
            if done % 10 == 0 or done == total_configs:
                print(
                    f"  {name}: {done}/{total_configs} configs processed "
                    f"(latest mean |dv|/v={metrics.mean_percent_error:.1f}%)"
                )

    rows.sort(key=lambda r: (r["name"], r["mean_abs"]))
    table = []
    for name in TARGETS:
        subset = [r for r in rows if r["name"] == name]
        subset.sort(key=lambda r: r["mean_abs"])  # best first
        for entry in subset[:5]:
            table.append(
                (
                    entry["name"],
                    entry["halo_v"],
                    entry["halo_r"],
                    entry["gas_scale"],
                    entry["gas_r"],
                    entry["bulge_scale"],
                    entry["kappa"],
                    entry["mean_abs"],
                    entry["max_abs"],
                )
            )

    header = [
        "Galaxy",
        "halo_v",
        "halo_r",
        "gas_scale",
        "gas_r",
        "bulge_scale",
        "kappa",
        "Mean|dv|/v [%]",
        "Max|dv|/v [%]",
    ]
    print(tabulate(table, headers=header, tablefmt="github"))


if __name__ == "__main__":
    main()
