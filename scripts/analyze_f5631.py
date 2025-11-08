"""Detailed diagnostics for SPARC galaxy F563-1 using rotmod data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics

TARGET = "F563-1"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    parser = SPARCParser(data_dir / "table1.dat", data_dir / "table2.dat")
    config = CSGConfig()
    model = CSGV4Model(config)

    record = parser.load_record(TARGET)
    galaxy = parser.to_galaxy_data(record, config)
    scan, residual_map = scan_kappa([galaxy], model=model, store_profiles=True)
    best_kappa = scan.best_kappa
    outputs = model.predict_velocity(galaxy, best_kappa)
    metrics = compute_error_metrics(galaxy, outputs)

    residuals = residual_map[TARGET]
    radii = galaxy.radii_kpc
    mean_inner = float(np.mean(residuals[radii <= np.median(radii)]))
    mean_outer = float(np.mean(residuals[radii > np.median(radii)]))

    rows = [
        (TARGET, galaxy.galaxy_type, best_kappa, metrics.mean_percent_error, metrics.max_percent_error),
    ]
    print(tabulate(rows, headers=["Galaxy", "Type", "Best kappa_c", "Mean |dv|/v [%]", "Max |dv|/v [%]"], tablefmt="github"))
    print(f"Inner residual mean: {mean_inner:.3f}, outer residual mean: {mean_outer:.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(radii, galaxy.v_obs, "o", label="Observed", linewidth=2.0)
    ax.plot(radii, outputs.v_pred, "-", label="Predicted", linewidth=2.0)
    v_baryonic = np.sqrt(record.v_gas**2 + record.v_disk**2 + record.v_bulge**2)
    ax.plot(radii, v_baryonic, "--", label="Baryonic", linewidth=1.5)
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title(f"F563-1 rotation curve (kappa_c={best_kappa:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "f5631_extended_rotation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
