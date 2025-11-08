"""Inspect CSG V4 booking invariants for SPARC galaxies."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
from tabulate import tabulate

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model


def summarize_entries(entries: List[Dict[str, float]]) -> Dict[str, float]:
    if not entries:
        return {}
    keys = entries[0].keys()
    return {key: mean(entry[key] for entry in entries) for key in keys}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    config = CSGConfig()
    parser = SPARCParser(table1, table2)
    galaxies = parser.load_galaxies(selection=None, config=config, stellar_ml=0.5)

    model = CSGV4Model(config)

    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    per_galaxy_rows = []

    for galaxy in galaxies.values():
        salience = model.compute_salience(galaxy)

        entry = {
            "R_galaxy": float(salience["R_galaxy"]),
            "phi_galaxy": float(salience["phi_galaxy"]),
            "S_h_galaxy": float(salience["S_h_galaxy"]),
            "K_galaxy": float(salience["K_galaxy"]),
            "J_galaxy": float(salience["J_galaxy"]),
            "PENALTY": float(salience["PENALTY"]),
            "A_galaxy": float(salience["A_galaxy"]),
            "CORE_mean": float(np.mean(salience["CORE"])),
            "HYPE_mean": float(np.mean(salience["HYPE"])),
            "Q_local_mean": float(np.mean(salience["Q_local"])),
            "S_prime_mean": float(np.mean(salience["S_prime"])),
            "C_local_mean": float(np.mean(salience["C_local"])),
            "W_local_mean": float(np.mean(salience["W_local"])),
            "M_local_mean": float(np.mean(salience["M_local"])),
        }

        grouped[galaxy.galaxy_type].append(entry)

        per_galaxy_rows.append([
            galaxy.name,
            galaxy.galaxy_type,
            f"{entry['R_galaxy']:.3f}",
            f"{entry['phi_galaxy']:.3f}",
            f"{entry['K_galaxy']:.3f}",
            f"{entry['J_galaxy']:.3f}",
            f"{entry['CORE_mean']:.3f}",
            f"{entry['HYPE_mean']:.3f}",
            f"{entry['PENALTY']:.3f}",
            f"{entry['S_prime_mean']:.4e}",
        ])

    headers = [
        "Galaxy",
        "Type",
        "R",
        "phi",
        "K",
        "J",
        "CORE",
        "HYPE",
        "Penalty",
        "S' mean",
    ]
    print("Per-galaxy booking indicators:")
    print(tabulate(per_galaxy_rows, headers=headers, tablefmt="github"))
    print()

    summary_rows = []
    for gal_type, entries in grouped.items():
        summary = summarize_entries(entries)
        summary_rows.append([
            gal_type,
            len(entries),
            f"{summary['R_galaxy']:.3f}",
            f"{summary['phi_galaxy']:.3f}",
            f"{summary['S_h_galaxy']:.3f}",
            f"{summary['K_galaxy']:.3f}",
            f"{summary['J_galaxy']:.3f}",
            f"{summary['CORE_mean']:.3f}",
            f"{summary['HYPE_mean']:.3f}",
            f"{summary['PENALTY']:.3f}",
            f"{summary['Q_local_mean']:.3f}",
            f"{summary['M_local_mean']:.3f}",
            f"{summary['W_local_mean']:.3f}",
        ])

    summary_headers = [
        "Type",
        "N",
        "R",
        "phi",
        "S_h",
        "K",
        "J",
        "CORE",
        "HYPE",
        "Penalty",
        "Q_local",
        "M_local",
        "W_local",
    ]

    print("\nGrouped means by galaxy type:")
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="github"))


if __name__ == "__main__":
    main()
