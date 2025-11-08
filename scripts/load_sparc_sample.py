"""Utility script to inspect SPARC rotation-curve tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    parser = SPARCParser(table1, table2)
    galaxies = parser.list_galaxies()
    print(f"Loaded SPARC metadata for {len(galaxies)} galaxies.")

    config = CSGConfig()
    sample = galaxies[:5]
    print("Sample galaxies:", ", ".join(sample))

    total_points = 0
    for name in sample:
        record = parser.load_record(name)
        galaxy = parser.to_galaxy_data(record, config)
        total_points += galaxy.n_radii
        print(f"- {name}: {galaxy.n_radii} radii, v_obs range {np.min(galaxy.v_obs):.1f}-{np.max(galaxy.v_obs):.1f} km/s")

    print(f"Total sample datapoints: {total_points}")


if __name__ == "__main__":
    main()
