"""Extract SPARC rotmod/surface brightness data for target galaxies."""

from __future__ import annotations

from pathlib import Path
import zipfile

TARGETS = ["F563-1", "F568-1", "F568-3", "F571-8", "IC4202"]


def extract_selected(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            stem = Path(member).stem
            if any(stem.startswith(target) for target in TARGETS):
                print(f"Extracting {member} from {zip_path.name}")
                zf.extract(member, path=out_dir)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    extended_dir = root / "data" / "sparc" / "extended"
    extract_dir = extended_dir / "extracted"

    extract_selected(extended_dir / "Rotmod_LTG.zip", extract_dir / "rotmod")
    extract_selected(extended_dir / "sfb_LTG.zip", extract_dir / "sfb")
    extract_selected(extended_dir / "BulgeDiskDec_LTG.zip", extract_dir / "bulge")


if __name__ == "__main__":
    main()
