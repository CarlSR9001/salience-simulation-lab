"""Reconstruct baryonic vs observed vs inferred dark profiles for outlier spirals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.optimizer import scan_kappa
from csg_v4.model import CSGV4Model

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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    rows = []
    for name in TARGETS:
        record = parser.load_record(name)
        galaxy = parser.to_galaxy_data(
            record,
            config,
            stellar_ml=DEFAULT_DISK_ML,
            stellar_ml_map=DISK_ML_MAP,
            bulge_ml=DEFAULT_BULGE_ML,
            bulge_ml_map=BULGE_ML_MAP,
        )

        scan, _ = scan_kappa([galaxy], model=model, store_profiles=True)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)

        v_obs = galaxy.v_obs
        v_bar = galaxy.v_bar
        v_dark = np.sqrt(np.clip(v_obs**2 - v_bar**2, 0.0, None))

        outer_idx = np.argmax(galaxy.radii_kpc)
        rows.append(
            (
                galaxy.name,
                galaxy.galaxy_type,
                best_kappa,
                float(galaxy.radii_kpc[outer_idx]),
                float(v_obs[outer_idx]),
                float(v_bar[outer_idx]),
                float(v_dark[outer_idx]),
            )
        )

        profile_path = root / "artifacts" / f"dark_profile_{galaxy.name}.csv"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            profile_path,
            np.column_stack((
                galaxy.radii_kpc,
                v_obs,
                v_bar,
                v_dark,
            )),
            delimiter=",",
            header="radius_kpc,v_obs_km_s,v_bar_km_s,v_dark_km_s",
            comments="",
            fmt="%.6f",
        )

    print(
        tabulate(
            rows,
            headers=[
                "Galaxy",
                "Type",
                "kappa_c",
                "R_max [kpc]",
                "v_obs [km/s]",
                "v_bar [km/s]",
                "v_dark [km/s]",
            ],
            tablefmt="github",
        )
    )
