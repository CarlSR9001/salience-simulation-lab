"""Scan kappa_c for each galaxy individually to assess universality drift."""

from __future__ import annotations

from pathlib import Path

from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.analysis import load_synthetic_galaxies
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa
from csg_v4.reporting import compute_error_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    config = CSGConfig()
    model = CSGV4Model(config)

    parser = SPARCParser(table1, table2)
    sparc_galaxies = parser.load_galaxies(selection=None, config=config)

    # SPARC quality flag 1 only
    q1_names = parser.table1[parser.table1["Q"] == 1].index.tolist()
    sparc_quality = [sparc_galaxies[name] for name in q1_names if name in sparc_galaxies]

    synthetic = list(load_synthetic_galaxies().values())

    rows = []
    for galaxy in sparc_quality + synthetic:
        scan, _ = scan_kappa([galaxy], model=model, store_profiles=False)
        best_kappa = scan.best_kappa
        outputs = model.predict_velocity(galaxy, best_kappa)
        metrics = compute_error_metrics(galaxy, outputs)
        rows.append(
            [
                galaxy.name,
                galaxy.galaxy_type,
                metrics.n_points,
                best_kappa,
                metrics.mean_percent_error,
                metrics.max_percent_error,
            ]
        )

    headers = ["Galaxy", "Type", "N", "Best kappa_c", "Mean |dv|/v [%]", "Max |dv|/v [%]"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
