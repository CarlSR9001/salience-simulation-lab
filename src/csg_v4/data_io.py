"""Data ingestion utilities for real galaxy datasets (SPARC)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CSGConfig
from .galaxy import GalaxyData


TABLE1_COLUMNS = [
    "Galaxy",
    "T",
    "D",
    "e_D",
    "f_D",
    "Inc",
    "e_Inc",
    "L36",
    "e_L36",
    "Reff",
    "SBeff",
    "Rdisk",
    "SBdisk",
    "MHI",
    "RHI",
    "Vflat",
    "e_Vflat",
    "Q",
    "Ref",
]


TABLE2_COLUMNS = [
    "Galaxy",
    "D",
    "radius_kpc",
    "v_obs",
    "e_v_obs",
    "v_gas",
    "v_disk",
    "v_bulge",
    "sb_disk",
    "sb_bulge",
]


@dataclass(slots=True)
class SPARCRecord:
    galaxy: str
    radius_kpc: np.ndarray
    v_obs: np.ndarray
    e_v_obs: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bulge: np.ndarray
    gas_fraction: float
    galaxy_type: str
    distance_mpc: float
    incl_deg: float


class SPARCParser:
    """Parser for the SPARC (Lelli+2016) tables from VizieR."""

    def __init__(self, table1_path: Path, table2_path: Path, rotmod_dir: Path | None = None) -> None:
        self.table1_path = Path(table1_path)
        self.table2_path = Path(table2_path)

        # Validate table1 path
        if not self.table1_path.exists():
            raise FileNotFoundError(
                f"SPARC table1 not found at: {self.table1_path}\n"
                f"Expected location: data/sparc/table1.dat\n"
                f"Please ensure you have downloaded the SPARC dataset."
            )
        if not self.table1_path.is_file():
            raise ValueError(f"table1 path is not a file: {self.table1_path}")
        if self.table1_path.stat().st_size == 0:
            raise ValueError(f"table1 file is empty: {self.table1_path}")

        # Validate table2 path
        if not self.table2_path.exists():
            raise FileNotFoundError(
                f"SPARC table2 not found at: {self.table2_path}\n"
                f"Expected location: data/sparc/table2.dat\n"
                f"Please ensure you have downloaded the SPARC dataset."
            )
        if not self.table2_path.is_file():
            raise ValueError(f"table2 path is not a file: {self.table2_path}")
        if self.table2_path.stat().st_size == 0:
            raise ValueError(f"table2 file is empty: {self.table2_path}")

        # Load tables with error handling
        try:
            self.table1 = self._load_table1(self.table1_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load SPARC table1 from {self.table1_path}.\n"
                f"Error: {e}\n"
                f"The file may be corrupted or in an unexpected format."
            ) from e

        try:
            self.table2 = self._load_table2(self.table2_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load SPARC table2 from {self.table2_path}.\n"
                f"Error: {e}\n"
                f"The file may be corrupted or in an unexpected format."
            ) from e

        # Validate that tables loaded successfully
        if self.table1.empty:
            raise ValueError(f"table1 loaded but contains no data: {self.table1_path}")
        if self.table2.empty:
            raise ValueError(f"table2 loaded but contains no data: {self.table2_path}")

        self.rotmod_dir = self._resolve_rotmod_dir(rotmod_dir)

    def _resolve_rotmod_dir(self, rotmod_dir: Path | None) -> Path | None:
        if rotmod_dir is not None:
            path = Path(rotmod_dir)
            return path if path.exists() else None
        candidate = self.table1_path.parent / "extended" / "extracted" / "rotmod"
        return candidate if candidate.exists() else None

    @staticmethod
    def _load_table1(path: Path) -> pd.DataFrame:
        df = pd.read_fwf(path, names=TABLE1_COLUMNS)
        numeric_cols = [
            "T",
            "D",
            "e_D",
            "f_D",
            "Inc",
            "e_Inc",
            "L36",
            "e_L36",
            "Reff",
            "SBeff",
            "Rdisk",
            "SBdisk",
            "MHI",
            "RHI",
            "Vflat",
            "e_Vflat",
            "Q",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Galaxy"] = df["Galaxy"].astype(str).str.strip()
        df.set_index("Galaxy", inplace=True)
        return df

    @staticmethod
    def _load_table2(path: Path) -> pd.DataFrame:
        df = pd.read_fwf(path, names=TABLE2_COLUMNS)
        numeric_cols = [
            "D",
            "radius_kpc",
            "v_obs",
            "e_v_obs",
            "v_gas",
            "v_disk",
            "v_bulge",
            "sb_disk",
            "sb_bulge",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Galaxy"] = df["Galaxy"].astype(str).str.strip()
        df["Galaxy"].replace("", np.nan, inplace=True)
        df["Galaxy"].ffill(inplace=True)
        return df

    def list_galaxies(self) -> list[str]:
        return sorted(self.table1.index.tolist())

    def _classify_type(self, hubble_t: int) -> str:
        if hubble_t <= 2:
            return "grand_design"
        if hubble_t <= 5:
            return "hsb_spiral"
        if hubble_t <= 8:
            return "spiral"
        if hubble_t <= 10:
            return "lsb_dwarf"
        return "dwarf_irregular"

    def load_record(self, galaxy: str) -> SPARCRecord:
        # Validate input
        if not galaxy or not isinstance(galaxy, str):
            raise ValueError(f"Galaxy name must be a non-empty string, got: {galaxy!r}")

        # Check if galaxy exists
        if galaxy not in self.table1.index:
            available = self.list_galaxies()
            close_matches = [g for g in available if galaxy.lower() in g.lower() or g.lower() in galaxy.lower()]
            error_msg = f"Galaxy '{galaxy}' not found in SPARC table1.\n"
            if close_matches:
                error_msg += f"Did you mean one of these? {', '.join(close_matches[:5])}\n"
            error_msg += f"Total available galaxies: {len(available)}\n"
            error_msg += f"Use list_galaxies() to see all available galaxy names."
            raise KeyError(error_msg)

        # Load rotation curve data
        curve = self.table2[self.table2["Galaxy"] == galaxy].copy()
        curve.dropna(subset=["radius_kpc", "v_obs"], inplace=True)
        curve = curve[np.isfinite(curve["radius_kpc"]) & np.isfinite(curve["v_obs"])]
        curve = curve[curve["radius_kpc"] > 0]
        curve = curve[curve["v_obs"] >= 0]
        curve.sort_values("radius_kpc", inplace=True)

        # Validate sufficient data points
        if curve.empty or curve.shape[0] < 3:
            raise ValueError(
                f"Galaxy '{galaxy}' has insufficient rotation-curve samples (n={curve.shape[0]}).\n"
                f"At least 3 valid data points are required for analysis.\n"
                f"This galaxy may have poor data quality or incomplete observations."
            )

        meta = self.table1.loc[galaxy]
        radius = curve["radius_kpc"].to_numpy(dtype=float)
        v_obs = curve["v_obs"].to_numpy(dtype=float)
        e_v_obs = curve["e_v_obs"].to_numpy(dtype=float)
        v_gas = curve["v_gas"].to_numpy(dtype=float)
        v_disk = curve["v_disk"].to_numpy(dtype=float)
        v_bulge = curve["v_bulge"].to_numpy(dtype=float)

        rotmod = self._load_rotmod_curve(galaxy)
        if rotmod is not None:
            radius = rotmod["radius"]
            v_obs = rotmod["v_obs"]
            e_v_obs = rotmod["sigma"]
            v_gas = rotmod["v_gas"]
            v_disk = rotmod["v_disk"]
            v_bulge = rotmod["v_bulge"]

        e_v_obs = np.nan_to_num(e_v_obs, nan=5.0, posinf=5.0, neginf=5.0)
        v_gas = np.nan_to_num(v_gas)
        v_disk = np.nan_to_num(v_disk)
        v_bulge = np.nan_to_num(v_bulge)

        sigma_v = np.maximum(e_v_obs, 0.5)

        mhi = float(meta.get("MHI", np.nan))
        l36 = float(meta.get("L36", np.nan))
        numerator = np.nan_to_num(mhi, nan=0.0)
        denominator = numerator + np.nan_to_num(l36, nan=0.0) + 1e-6
        gas_fraction = float(np.clip(numerator / denominator if denominator > 0 else 0.0, 0.0, 1.0))

        record = SPARCRecord(
            galaxy=galaxy,
            radius_kpc=radius,
            v_obs=v_obs,
            e_v_obs=sigma_v,
            v_gas=v_gas,
            v_disk=v_disk,
            v_bulge=v_bulge,
            gas_fraction=gas_fraction,
            galaxy_type=self._classify_type(int(meta["T"])),
            distance_mpc=float(meta["D"]),
            incl_deg=float(meta["Inc"]),
        )
        return record

    def _load_rotmod_curve(self, galaxy: str) -> dict[str, np.ndarray] | None:
        if self.rotmod_dir is None:
            return None
        path = self.rotmod_dir / f"{galaxy}_rotmod.dat"
        if not path.exists():
            return None
        try:
            data = np.loadtxt(path, comments="#")
        except OSError:
            return None
        if data.ndim != 2 or data.shape[1] < 6:
            return None
        radius = data[:, 0].astype(float)
        v_obs = data[:, 1].astype(float)
        sigma = data[:, 2].astype(float)
        v_gas = data[:, 3].astype(float)
        v_disk = data[:, 4].astype(float)
        v_bulge = data[:, 5].astype(float)
        mask = (radius > 0) & np.isfinite(v_obs)
        if not np.any(mask):
            return None
        return {
            "radius": radius[mask],
            "v_obs": v_obs[mask],
            "sigma": sigma[mask],
            "v_gas": v_gas[mask],
            "v_disk": v_disk[mask],
            "v_bulge": v_bulge[mask],
        }

    def to_galaxy_data(
        self,
        record: SPARCRecord,
        config: CSGConfig,
        stellar_ml: float = 0.5,
        stellar_ml_map: Mapping[str, float] | None = None,
        bulge_ml: float = 0.7,
        bulge_ml_map: Mapping[str, float] | None = None,
    ) -> GalaxyData:
        # Validate mass-to-light ratios
        if not (0.0 < stellar_ml < 10.0):
            raise ValueError(
                f"stellar_ml must be in range (0, 10), got {stellar_ml}.\n"
                f"Typical values for stellar M/L are 0.3-1.5 in solar units."
            )
        if not (0.0 < bulge_ml < 10.0):
            raise ValueError(
                f"bulge_ml must be in range (0, 10), got {bulge_ml}.\n"
                f"Typical values for bulge M/L are 0.5-2.0 in solar units."
            )

        ml_disk = stellar_ml_map.get(record.galaxy_type, stellar_ml) if stellar_ml_map else stellar_ml
        ml_bulge = bulge_ml_map.get(record.galaxy_type, bulge_ml) if bulge_ml_map else bulge_ml

        # Validate mapped M/L values
        if not (0.0 < ml_disk < 10.0):
            raise ValueError(
                f"Mapped disk M/L for {record.galaxy_type} is {ml_disk}, must be in range (0, 10)."
            )
        if not (0.0 < ml_bulge < 10.0):
            raise ValueError(
                f"Mapped bulge M/L for {record.galaxy_type} is {ml_bulge}, must be in range (0, 10)."
            )

        v_disk_scaled = record.v_disk * np.sqrt(ml_disk)
        v_bulge_scaled = record.v_bulge * np.sqrt(ml_bulge)
        v_stellar = np.sqrt(np.square(v_disk_scaled) + np.square(v_bulge_scaled))
        v_gas = np.abs(record.v_gas)
        v_bar = np.sqrt(np.square(v_stellar) + np.square(v_gas))
        sigma_v = np.maximum(record.e_v_obs, 0.5)

        galaxy = GalaxyData(
            name=record.galaxy,
            galaxy_type=record.galaxy_type,
            radii_kpc=record.radius_kpc,
            v_obs=record.v_obs,
            v_bar=v_bar,
            sigma_v=sigma_v,
            gas_fraction=record.gas_fraction,
            age_gyr=10.0,
            has_coherent_rotation=True,
            metadata={
                "distance_mpc": record.distance_mpc,
                "incl_deg": record.incl_deg,
                "stellar_ml_disk": ml_disk,
                "stellar_ml_bulge": ml_bulge,
            },
        )
        return galaxy

    def load_galaxies(
        self,
        selection: Iterable[str] | None,
        config: CSGConfig,
        stellar_ml: float = 0.5,
        stellar_ml_map: Mapping[str, float] | None = None,
        bulge_ml: float = 0.7,
        bulge_ml_map: Mapping[str, float] | None = None,
    ) -> dict[str, GalaxyData]:
        # Validate config
        if not isinstance(config, CSGConfig):
            raise TypeError(f"config must be a CSGConfig instance, got {type(config)}")

        # Get galaxy list
        if selection is None:
            galaxies = self.list_galaxies()
        else:
            galaxies = list(selection)
            if not galaxies:
                raise ValueError("selection is empty - no galaxies to load")

        result: dict[str, GalaxyData] = {}
        skipped: list[str] = []
        errors: list[tuple[str, str]] = []

        for gal in galaxies:
            try:
                record = self.load_record(gal)
            except (ValueError, KeyError) as e:
                skipped.append(gal)
                errors.append((gal, str(e).split('\n')[0]))  # First line of error
                continue
            except Exception as e:
                skipped.append(gal)
                errors.append((gal, f"Unexpected error: {e}"))
                continue

            try:
                result[gal] = self.to_galaxy_data(
                    record,
                    config,
                    stellar_ml=stellar_ml,
                    stellar_ml_map=stellar_ml_map,
                    bulge_ml=bulge_ml,
                    bulge_ml_map=bulge_ml_map,
                )
            except Exception as e:
                skipped.append(gal)
                errors.append((gal, f"Conversion error: {e}"))
                continue

        # Report results
        if skipped:
            print(f"[SPARCParser] Skipped {len(skipped)} galaxies lacking valid data: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")
            if len(errors) <= 5:
                for gal, err in errors:
                    print(f"  - {gal}: {err}")

        if not result:
            raise ValueError(
                f"No galaxies could be loaded successfully.\n"
                f"Attempted to load {len(galaxies)} galaxies, all failed.\n"
                f"First few errors:\n" + "\n".join(f"  - {gal}: {err}" for gal, err in errors[:3])
            )

        return result
