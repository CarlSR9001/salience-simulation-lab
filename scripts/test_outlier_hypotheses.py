"""Experiment with outer-disk hypotheses for SPARC spiral outliers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable

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

HALO_VELOCITY = 80.0  # km/s
HALO_RADIUS_THRESHOLD = 8.0  # kpc
GAS_SCALE = 1.6
GAS_RADIUS_THRESHOLD = 6.0  # kpc
BULGE_SCALE = 1.6


def build_galaxy(record, config: CSGConfig, v_bar: np.ndarray, sigma: np.ndarray) -> GalaxyData:
    return GalaxyData(
        name=record.galaxy,
        galaxy_type=record.galaxy_type,
        radii_kpc=record.radius_kpc,
        v_obs=record.v_obs,
        v_bar=v_bar,
        sigma_v=sigma,
        gas_fraction=record.gas_fraction,
        age_gyr=10.0,
        has_coherent_rotation=True,
        metadata={
            "distance_mpc": record.distance_mpc,
            "incl_deg": record.incl_deg,
        },
    )


def construct_variants(record, config: CSGConfig) -> Dict[str, GalaxyData]:
    disk_ml = DISK_ML_MAP.get(record.galaxy_type, DEFAULT_DISK_ML)
    bulge_ml = BULGE_ML_MAP.get(record.galaxy_type, DEFAULT_BULGE_ML)

    v_disk = record.v_disk * np.sqrt(disk_ml)
    v_bulge = record.v_bulge * np.sqrt(bulge_ml)
    v_gas = np.abs(record.v_gas)
    sigma = np.maximum(record.e_v_obs, 0.5)

    base = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)

    variants: Dict[str, GalaxyData] = {
        "baseline": build_galaxy(record, config, base, sigma),
    }

    radii = record.radius_kpc

    # Hypothesis A: add flat halo tail
    halo_component = np.zeros_like(radii)
    halo_component[radii >= HALO_RADIUS_THRESHOLD] = HALO_VELOCITY
    v_halo = halo_component
    halo_v_bar = np.sqrt(base**2 + v_halo**2)
    variants["halo_flat"] = build_galaxy(record, config, halo_v_bar, sigma)

    # Hypothesis B: extended gas disk
    gas_scaled = v_gas.copy()
    gas_scaled[radii >= GAS_RADIUS_THRESHOLD] *= GAS_SCALE
    gas_v_bar = np.sqrt(v_disk**2 + v_bulge**2 + gas_scaled**2)
    variants["extended_gas"] = build_galaxy(record, config, gas_v_bar, sigma)

    # Hypothesis C: heavy bulge continuation
    bulge_scaled = v_bulge * BULGE_SCALE
    bulge_v_bar = np.sqrt(v_disk**2 + bulge_scaled**2 + v_gas**2)
    variants["bulge_heavy"] = build_galaxy(record, config, bulge_v_bar, sigma)

    return variants


def analyze_variants(galaxies: Iterable[GalaxyData], model: CSGV4Model):
    rows = []
    for galaxy in galaxies:
        scan, _ = scan_kappa([galaxy], model=model, store_profiles=False)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)
        rows.append(
            (
                galaxy.metadata.get("scenario", "baseline"),
                best_kappa,
                metrics.mean_percent_error,
                metrics.max_percent_error,
            )
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    results = []
    for name in TARGETS:
        record = parser.load_record(name)
        variants = construct_variants(record, config)
        annotated_variants = []
        for scenario, galaxy in variants.items():
            galaxy.metadata = dict(galaxy.metadata or {})
            galaxy.metadata["scenario"] = scenario
            annotated_variants.append(galaxy)
        rows = analyze_variants(annotated_variants, model)
        for scenario, kappa, mean_abs, max_abs in rows:
            results.append((name, scenario, kappa, mean_abs, max_abs))

    print(tabulate(results, headers=["Galaxy", "Scenario", "Best kappa", "Mean |dv|/v [%]", "Max |dv|/v [%]"], tablefmt="github"))


if __name__ == "__main__":
    main()
