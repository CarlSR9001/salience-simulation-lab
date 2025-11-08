"""Analyze SPARC outliers using detailed rotmod mass components."""

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

TARGETS = ["F563-1", "F568-1", "F568-3", "F571-8", "IC4202"]


def load_rotmod(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    radii = data[:, 0]
    v_obs = data[:, 1]
    sigma = data[:, 2]
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    v_bulge = data[:, 5]
    v_bar = np.sqrt(np.clip(v_gas**2 + v_disk**2 + v_bulge**2, 0.0, None))
    return {
        "radii": radii,
        "v_obs": v_obs,
        "sigma": np.maximum(sigma, 0.5),
        "v_bar": v_bar,
    }


def build_galaxy(name: str, parser: SPARCParser, rotmod_dir: Path) -> GalaxyData:
    record = parser.table1.loc[name]
    rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
    components = load_rotmod(rotmod_path)

    distance = float(record.get("Dist", np.nan)) if "Dist" in record.index else np.nan
    incl = float(record.get("i", np.nan)) if "i" in record.index else np.nan
    metadata = {
        "distance_mpc": distance,
        "incl_deg": incl,
        "scenario": "rotmod_ext",
    }

    return GalaxyData(
        name=name,
        galaxy_type=str(record.get("Type", "unknown")),
        radii_kpc=components["radii"],
        v_obs=components["v_obs"],
        v_bar=components["v_bar"],
        sigma_v=components["sigma"],
        gas_fraction=float(record.get("fgas", 0.0)),
        age_gyr=10.0,
        has_coherent_rotation=True,
        metadata=metadata,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    extended_dir = data_dir / "extended" / "extracted"
    rotmod_dir = extended_dir / "rotmod"

    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    rows = []
    for name in TARGETS:
        galaxy = build_galaxy(name, parser, rotmod_dir)
        scan, _ = scan_kappa([galaxy], model=model, store_profiles=False)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)
        rows.append(
            (
                name,
                galaxy.galaxy_type,
                best_kappa,
                metrics.mean_percent_error,
                metrics.max_percent_error,
            )
        )

        out_path = root / "artifacts" / f"extended_profile_{name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_path,
            np.column_stack(
                (
                    galaxy.radii_kpc,
                    galaxy.v_obs,
                    galaxy.v_bar,
                    outputs.v_pred,
                )
            ),
            delimiter=",",
            header="radius_kpc,v_obs_km_s,v_bar_ext_km_s,v_pred_km_s",
            comments="",
            fmt="%.6f",
        )

    print(
        tabulate(
            rows,
            headers=["Galaxy", "Type", "Best kappa_c", "Mean |dv|/v [%]", "Max |dv|/v [%]"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
