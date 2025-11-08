"""Run CSG V4 analysis across the entire SPARC catalogue."""

from __future__ import annotations

from pathlib import Path
import argparse
import time

import numpy as np
from tqdm import tqdm

from csg_v4.analysis import CSGAnalysis, load_synthetic_galaxies
from csg_v4.config import CSGConfig
from csg_v4.data_io import SPARCParser
from csg_v4.model import CSGV4Model
from csg_v4.optimizer import scan_kappa


def filter_by_quality(galaxies, metadata_table, max_quality: int | None):
    if max_quality is None:
        return galaxies
    filtered = []
    for galaxy in galaxies:
        if galaxy.name not in metadata_table.index:
            continue
        quality = int(metadata_table.loc[galaxy.name, "Q"])
        if quality <= max_quality:
            filtered.append(galaxy)
    return filtered


def apply_feedback(galaxies, residual_map):
    adjustments = []
    face_threshold = -0.3
    heel_threshold = 0.3
    neutral_band = 0.15
    smoothing = 0.6
    for galaxy in galaxies:
        resid = residual_map.get(galaxy.name)
        if resid is None or resid.size == 0:
            continue
        mean_resid = float(np.mean(resid))
        abs_mean = float(np.mean(np.abs(resid)))

        metadata = dict(galaxy.metadata or {})
        residual_avg = float(metadata.get("residual_avg", mean_resid))
        residual_avg = smoothing * residual_avg + (1.0 - smoothing) * mean_resid

        prev_role = str(metadata.get("crowd_role", "tweener"))
        role = prev_role
        if residual_avg <= face_threshold - neutral_band:
            role = "face"
        elif residual_avg >= heel_threshold + neutral_band:
            role = "heel"
        elif abs(residual_avg) <= neutral_band:
            role = "tweener"

        hype_delta = 0.0
        phi_delta = 0.0
        w_scale = 1.0

        magnitude = abs(residual_avg)
        if role == "face":
            hype_delta += min(0.6, 0.2 + 0.5 * magnitude)
            phi_delta -= min(0.12, 0.04 + 0.08 * magnitude)
            w_scale *= max(0.55, 1.0 - 0.5 * magnitude)
        elif role == "heel":
            hype_delta -= min(0.5, 0.15 + 0.45 * magnitude)
            phi_delta += min(0.12, 0.04 + 0.07 * magnitude)
            w_scale *= min(1.6, 1.0 + 0.5 * magnitude)
        else:  # tweener
            hype_delta += np.clip(mean_resid, -0.1, 0.1)
            w_scale *= np.clip(1.0 + 0.3 * mean_resid, 0.8, 1.3)

        metadata["feedback_hype"] = hype_delta
        metadata["feedback_phi_delta"] = phi_delta
        metadata["feedback_w_scale"] = w_scale
        metadata["mean_residual"] = mean_resid
        metadata["residual_avg"] = residual_avg
        metadata["crowd_role"] = role
        galaxy.metadata = metadata
        adjustments.append(
            (
                galaxy.name,
                galaxy.galaxy_type.lower(),
                residual_avg,
                mean_resid,
                role,
                hype_delta,
                phi_delta,
                w_scale,
            )
        )
    return adjustments


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_header(text: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SPARC analysis with optional quality filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all SPARC galaxies
  %(prog)s --max-quality 1                    # Only highest quality
  %(prog)s --verbose                          # Show detailed progress
  %(prog)s --disk-ml 0.6 --bulge-ml 0.8       # Custom M/L ratios
  %(prog)s --include-synthetic                # Add synthetic galaxies
        """,
    )
    parser.add_argument("--max-quality", type=int, default=None, help="Maximum SPARC quality flag to include (1=best).")
    parser.add_argument(
        "--disk-ml",
        type=float,
        default=0.5,
        help="Default stellar M/L for disks (SPARC units, default: 0.5).",
    )
    parser.add_argument(
        "--bulge-ml",
        type=float,
        default=0.7,
        help="Default stellar M/L for bulges (SPARC units, default: 0.7).",
    )
    parser.add_argument(
        "--disk-ml-map",
        nargs="*",
        default=[],
        metavar="TYPE=VALUE",
        help="Override disk M/L per galaxy type (e.g., hsb_spiral=0.8 spiral=0.7).",
    )
    parser.add_argument(
        "--bulge-ml-map",
        nargs="*",
        default=[],
        metavar="TYPE=VALUE",
        help="Override bulge M/L per galaxy type.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Append synthetic benchmark galaxies to the analysis set.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors and final results.",
    )
    args = parser.parse_args()

    # Validate arguments
    import sys

    if args.max_quality is not None:
        if not (1 <= args.max_quality <= 3):
            print(
                f"Error: --max-quality must be 1, 2, or 3 (got {args.max_quality}).\n"
                f"  1 = highest quality (most reliable)\n"
                f"  2 = medium quality\n"
                f"  3 = lowest quality (include with caution)",
                file=sys.stderr
            )
            sys.exit(1)

    if not (0.0 < args.disk_ml < 10.0):
        print(
            f"Error: --disk-ml must be in range (0, 10), got {args.disk_ml}.\n"
            f"Typical values are 0.3-1.5 in solar units.",
            file=sys.stderr
        )
        sys.exit(1)

    if not (0.0 < args.bulge_ml < 10.0):
        print(
            f"Error: --bulge-ml must be in range (0, 10), got {args.bulge_ml}.\n"
            f"Typical values are 0.5-2.0 in solar units.",
            file=sys.stderr
        )
        sys.exit(1)

    # Set global verbosity level (used by other modules)
    import os
    if args.quiet:
        os.environ["CSG_VERBOSITY"] = "0"
    elif args.verbose:
        os.environ["CSG_VERBOSITY"] = "2"
    else:
        os.environ["CSG_VERBOSITY"] = "1"

    start_time = time.time()

    if not args.quiet:
        print_header("CSG V4 SPARC Galaxy Analysis")

    # Locate and validate data files
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "sparc"
    table1 = data_dir / "table1.dat"
    table2 = data_dir / "table2.dat"

    if not data_dir.exists():
        print(
            f"Error: SPARC data directory not found: {data_dir}\n"
            f"Expected location: {project_root}/data/sparc/\n"
            f"Please ensure the SPARC dataset has been downloaded and extracted.",
            file=sys.stderr
        )
        sys.exit(1)

    if not table1.exists():
        print(
            f"Error: SPARC table1 not found: {table1}\n"
            f"Expected location: {data_dir}/table1.dat\n"
            f"Please download the SPARC dataset from Lelli et al. (2016).",
            file=sys.stderr
        )
        sys.exit(1)

    if not table2.exists():
        print(
            f"Error: SPARC table2 not found: {table2}\n"
            f"Expected location: {data_dir}/table2.dat\n"
            f"Please download the SPARC dataset from Lelli et al. (2016).",
            file=sys.stderr
        )
        sys.exit(1)

    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Data directory: {data_dir}")
        print(f"  Max quality: {args.max_quality if args.max_quality else 'all'}")
        print(f"  Disk M/L: {args.disk_ml}")
        print(f"  Bulge M/L: {args.bulge_ml}")

    # Initialize SPARC parser
    try:
        sparc_parser = SPARCParser(table1, table2)
    except Exception as e:
        print(
            f"Error: Failed to initialize SPARC parser.\n"
            f"Details: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    config = CSGConfig()

    def parse_ml_map(entries):
        """Parse M/L map entries with validation."""
        mapping = {}
        invalid_entries = []
        for entry in entries:
            if "=" not in entry:
                invalid_entries.append(entry)
                continue
            key, value = entry.split("=", 1)
            try:
                float_value = float(value)
                if not (0.0 < float_value < 10.0):
                    print(
                        f"Warning: M/L value {float_value} for '{key.strip()}' is outside typical range (0, 10).",
                        file=sys.stderr
                    )
                mapping[key.strip()] = float_value
            except ValueError:
                invalid_entries.append(entry)
                continue
        if invalid_entries:
            print(
                f"Warning: Ignoring invalid M/L map entries: {', '.join(invalid_entries)}",
                file=sys.stderr
            )
        return mapping

    disk_ml_map = parse_ml_map(args.disk_ml_map)
    bulge_ml_map = parse_ml_map(args.bulge_ml_map)

    if not args.quiet:
        print("\nLoading SPARC galaxy data...")
    load_start = time.time()

    try:
        galaxies_dict = sparc_parser.load_galaxies(
            selection=None,
            config=config,
            stellar_ml=args.disk_ml,
            stellar_ml_map=disk_ml_map,
            bulge_ml=args.bulge_ml,
            bulge_ml_map=bulge_ml_map,
        )
    except Exception as e:
        print(
            f"Error: Failed to load SPARC galaxies.\n"
            f"Details: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    metadata_table = sparc_parser.table1
    sparc_galaxies = filter_by_quality(list(galaxies_dict.values()), metadata_table, args.max_quality)

    galaxies = list(sparc_galaxies)
    if args.include_synthetic:
        synthetic = list(load_synthetic_galaxies().values())
        galaxies.extend(synthetic)

    load_elapsed = time.time() - load_start

    if not args.quiet:
        if args.include_synthetic:
            print(
                f"Loaded {len(sparc_galaxies)} SPARC + {len(synthetic) if args.include_synthetic else 0} synthetic galaxies "
                f"with {sum(g.n_radii for g in galaxies)} total data points in {format_duration(load_elapsed)}"
            )
        else:
            print(
                f"Loaded {len(sparc_galaxies)} SPARC galaxies with {sum(g.n_radii for g in sparc_galaxies)} data points "
                f"in {format_duration(load_elapsed)}"
            )

    model = CSGV4Model(config)

    if not args.quiet:
        print("\n" + "-" * 70)
        print("Phase 1: Initial kappa scan")
        print("-" * 70)

    scan_start = time.time()
    initial_scan, residual_map = scan_kappa(galaxies, model=model, store_profiles=True)
    scan_elapsed = time.time() - scan_start

    if not args.quiet:
        print(f"\nInitial scan completed in {format_duration(scan_elapsed)}")
        print(f"Best kappa_c: {initial_scan.best_kappa:.6f} (chi^2 = {initial_scan.best_chisq:.6e})")

    if not args.quiet:
        print("\n" + "-" * 70)
        print("Phase 2: Applying feedback adjustments")
        print("-" * 70)

    feedback_start = time.time()
    adjustments = apply_feedback(galaxies, residual_map)
    feedback_elapsed = time.time() - feedback_start

    if adjustments:
        if not args.quiet:
            print(f"\nApplied feedback to {len(adjustments)} galaxies in {format_duration(feedback_elapsed)}")
        if args.verbose:
            print("\nDetailed feedback adjustments:")
            print(f"{'Name':<12} {'Type':<14} {'ResidAvg':<9} {'MeanRes':<9} {'Role':<10} {'Hype':<7} {'PhiΔ':<7} {'WScale':<7}")
            print("-" * 90)
            for name, gtype, residual_avg, mean_resid, role, hype_delta, phi_delta, w_scale in adjustments:
                print(
                    f"{name:<12} {gtype:<14} {residual_avg:+8.3f} {mean_resid:+8.3f} {role:<10} "
                    f"{hype_delta:+6.3f} {phi_delta:+6.3f} {w_scale:6.3f}"
                )
    else:
        if not args.quiet:
            print(f"\nNo feedback adjustments applied ({format_duration(feedback_elapsed)})")

    if not args.quiet:
        print("\n" + "-" * 70)
        print("Phase 3: Full CSG analysis")
        print("-" * 70)

    analysis_start = time.time()
    analysis = CSGAnalysis(config)
    artifacts = analysis.run(
        galaxies=galaxies,
        output_dir=project_root / "artifacts_sparc",
        plot_limit=25,
        surface_sample=18,
    )
    analysis_elapsed = time.time() - analysis_start

    total_elapsed = time.time() - start_time

    if args.quiet:
        # Minimal output
        print(f"{artifacts.scan_result.best_kappa:.6f}")
    else:
        print_header("Analysis Complete")

        print(f"\nTotal runtime: {format_duration(total_elapsed)}")
        print(f"  Loading:   {format_duration(load_elapsed)}")
        print(f"  Scanning:  {format_duration(scan_elapsed)}")
        print(f"  Feedback:  {format_duration(feedback_elapsed)}")
        print(f"  Analysis:  {format_duration(analysis_elapsed)}")

        print("\n--- Kappa Fit Results ---")
        print(f"  Best-fit kappa_c:           {artifacts.scan_result.best_kappa:.6f}")
        print(f"  Chi-squared:                {artifacts.scan_result.best_chisq:.6e}")

        if artifacts.weighted_best_kappa is not None:
            print(f"  Radius-weighted kappa_c:    {artifacts.weighted_best_kappa:.6f}")

        print(f"  Bootstrap mean:             {artifacts.bootstrap_kappas.mean():.6f}")
        print(f"  Bootstrap std:              {artifacts.bootstrap_kappas.std():.6f}")

        print("\n--- Output ---")
        print(f"  Directory: {(project_root / 'artifacts_sparc').resolve()}")
        print(f"  Plots: {len(artifacts.plot_targets)} rotation curves")

        if args.verbose:
            print("\n--- Per-Galaxy Metrics ---")
            print(artifacts.tables)
        else:
            print("\nRun with --verbose to see per-galaxy metrics")

        print()


if __name__ == "__main__":
    main()
