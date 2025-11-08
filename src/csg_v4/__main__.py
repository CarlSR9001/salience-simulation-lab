"""Command-line entry point for running the CSG V4 synthetic analysis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .analysis import run_synthetic_pipeline


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


def print_summary(artifacts, elapsed: float, verbose: bool) -> None:
    """Print analysis summary statistics."""
    print_header("Analysis Summary")

    print(f"\nRuntime: {format_duration(elapsed)}")
    print(f"Output Directory: {artifacts.config}")

    print("\n--- Kappa Fit Results ---")
    print(f"  Primary scan best kappa_c:  {artifacts.scan_result.best_kappa:.6f}")
    print(f"  Chi-squared:                {artifacts.scan_result.best_chisq:.6e}")

    if artifacts.weighted_best_kappa is not None:
        print(f"  Radius-weighted kappa_c:    {artifacts.weighted_best_kappa:.6f}")

    if artifacts.refinement.quadratic is not None:
        quad = artifacts.refinement.quadratic
        print(f"  Quadratic refinement:       {quad.kappa:.6f} (σ ≈ {quad.sigma:.4f})")

    if artifacts.refinement.golden is not None:
        gold = artifacts.refinement.golden
        print(f"  Golden-section refinement:  {gold.kappa:.6f} ({gold.iterations} iterations)")

    print(f"\n  Bootstrap mean:             {artifacts.bootstrap_kappas.mean():.6f}")
    print(f"  Bootstrap std:              {artifacts.bootstrap_kappas.std():.6f}")

    print("\n--- Galaxy Statistics ---")
    num_galaxies = len(artifacts.metrics)
    total_points = sum(len(artifacts.residual_map[name]) for name in artifacts.residual_map)
    print(f"  Galaxies analyzed:          {num_galaxies}")
    print(f"  Total data points:          {total_points}")
    print(f"  Rotation curves plotted:    {len(artifacts.plot_targets)}")

    if verbose:
        print("\n--- Per-Galaxy Metrics ---")
        print(artifacts.tables)


def validate_output_dir(output_dir: Path) -> None:
    """Validate that output directory can be created and is writable."""
    import sys

    # Check if path exists
    if output_dir.exists():
        if not output_dir.is_dir():
            print(
                f"Error: Output path exists but is not a directory: {output_dir}\n"
                f"Please specify a different path or remove the existing file.",
                file=sys.stderr
            )
            sys.exit(1)
    else:
        # Check if parent directory exists or can be created
        parent = output_dir.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(
                    f"Error: Cannot create parent directory: {parent}\n"
                    f"Permission denied. Please check your file system permissions.",
                    file=sys.stderr
                )
                sys.exit(1)
            except Exception as e:
                print(
                    f"Error: Failed to create parent directory: {parent}\n"
                    f"Details: {e}",
                    file=sys.stderr
                )
                sys.exit(1)


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(
        description="Run Continuity–Strain Gravity (CSG) V4 synthetic analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default settings
  %(prog)s --verbose                 # Show detailed output
  %(prog)s --output-dir results/     # Custom output directory
  %(prog)s --quiet                   # Minimal output
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where plots and metric summaries will be written (default: artifacts).",
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

    # Validate output directory before proceeding
    validate_output_dir(args.output_dir)

    # Set global verbosity level (used by other modules)
    import os
    if args.quiet:
        os.environ["CSG_VERBOSITY"] = "0"
    elif args.verbose:
        os.environ["CSG_VERBOSITY"] = "2"
    else:
        os.environ["CSG_VERBOSITY"] = "1"

    if not args.quiet:
        print_header("CSG V4 Synthetic Galaxy Analysis")
        print(f"\nOutput directory: {args.output_dir.resolve()}")
        print(f"Verbosity level: {'quiet' if args.quiet else 'verbose' if args.verbose else 'normal'}")

    start_time = time.time()

    try:
        artifacts = run_synthetic_pipeline(output_dir=args.output_dir)
        elapsed = time.time() - start_time

        if args.quiet:
            # Minimal output - just the key result
            print(f"{artifacts.scan_result.best_kappa:.6f}")
        else:
            # Normal or verbose output
            print_summary(artifacts, elapsed, args.verbose)

            if not args.verbose:
                print("\n" + artifacts.tables)
                print(f"\nBest-fit kappa_c: {artifacts.scan_result.best_kappa:.6f}")

            print_header("Analysis Complete")
            print(f"Results written to: {args.output_dir.resolve()}\n")

    except ImportError as e:
        elapsed = time.time() - start_time
        print(
            f"\nError after {format_duration(elapsed)}:\n"
            f"Missing required dependency: {e}\n"
            f"Please install required packages: pip install numpy pandas matplotlib",
            file=sys.stderr
        )
        sys.exit(1)
    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        print(
            f"\nError after {format_duration(elapsed)}:\n"
            f"File not found: {e}\n"
            f"Please check that all required data files are present.",
            file=sys.stderr
        )
        sys.exit(1)
    except ValueError as e:
        elapsed = time.time() - start_time
        print(
            f"\nError after {format_duration(elapsed)}:\n"
            f"Invalid input data: {e}",
            file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - start_time
        print(
            f"\nError after {format_duration(elapsed)}: {e}\n"
            f"For help, run: python -m csg_v4 --help",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
