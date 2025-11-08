# CSG V4 Usage Examples

This directory contains practical examples demonstrating how to use the CSG V4 package.

## Examples

### 1. `basic_usage.py`

**Purpose:** Learn the fundamental CSG V4 workflow

**What it demonstrates:**
- Creating a `GalaxyData` instance
- Initializing a `CSGV4Model`
- Computing salience profiles
- Predicting rotation curves
- Visualizing results

**Usage:**
```bash
cd docs/examples
python basic_usage.py
```

**Output:**
- `csg_v4_basic_example.png`: 4-panel figure showing rotation curves, salience, quality factors, and residuals

**Recommended for:** First-time users wanting a quick overview

---

### 2. `kappa_optimization.py`

**Purpose:** Learn how to optimize κ_c across multiple galaxies

**What it demonstrates:**
- Loading synthetic galaxy samples
- Grid scanning for optimal κ_c
- Refining estimates with quadratic fit and golden section search
- Estimating uncertainty via bootstrap resampling
- Visualizing scan results and galaxy fits

**Usage:**
```bash
cd docs/examples
python kappa_optimization.py
```

**Output:**
- `csg_v4_kappa_optimization.png`: Kappa scan curve and bootstrap distribution
- `csg_v4_galaxy_fits.png`: Rotation curve fits for each galaxy

**Recommended for:** Users fitting CSG V4 to their own galaxy samples

---

### 3. `custom_galaxy.py`

**Purpose:** Explore how galaxy properties affect salience

**What it demonstrates:**
- Creating custom galaxies with specific parameters
- Comparing salience across morphological types (grand design, spiral, dwarf, etc.)
- Parameter sensitivity analysis (dispersion, gas fraction, age)
- Interpreting scoring components

**Usage:**
```bash
cd docs/examples
python custom_galaxy.py
```

**Output:**
- `morphology_comparison.png`: Salience profiles for different galaxy types
- `parameter_sensitivity.png`: How salience changes with galaxy parameters

**Recommended for:** Users understanding how CSG V4 responds to different galaxy properties

---

## Running the Examples

All examples are standalone scripts that can be run directly:

```bash
# From the SAL directory
python -m docs.examples.basic_usage

# Or navigate to examples and run
cd docs/examples
python basic_usage.py
```

## Dependencies

Examples require:
- `numpy`
- `matplotlib`
- `csg_v4` (install with `pip install -e .` from SAL directory)

## Customization

Feel free to modify these examples for your own use cases:

- **Change parameters:** Edit `CSGConfig` initialization
- **Load real data:** Replace synthetic galaxies with SPARC data via `SPARCParser`
- **Adjust visualizations:** Modify matplotlib styling and layout
- **Add new analyses:** Use these as templates for your own scripts

## Getting Help

- **API reference:** See docstrings in `src/csg_v4/*.py`
- **Theory:** Read [`../CONCEPTS.md`](../CONCEPTS.md)
- **Main docs:** Check [`../../README.md`](../../README.md)

## Example Output Preview

After running `basic_usage.py`, you should see:

```
==============================================================
CSG V4 Basic Usage Example
==============================================================

[1] Creating example galaxy...
    Name: ExampleSpiral
    Type: hsb_spiral
    Radii: 25 points from 1.0 to 20.0 kpc
    Max velocity: 189.0 km/s

[2] Initializing CSG V4 model...
    Using a0 = 20955.5 kpc/Gyr²

[3] Computing salience profile...
    Mean S': 0.123
    Mean Q_local: 0.871
    Mean Q_final: 0.865
    Galaxy aura (A): 0.859

[4] Predicting rotation curve...
    Using κ_c = 0.5
    RMS error: 12.34 km/s
    Mean fractional error: 4.56%

[5] Creating visualization...
    Saved to: csg_v4_basic_example.png

[6] Exploring scoring components...
    R_galaxy (retention): 0.823
    K_galaxy (coherence): 0.950
    phi_galaxy (disorder): 0.400
    Mean C_local (continuity): 0.789
    Mean M_local (mass): 0.456
    Mean W_local (curvature): 0.234

==============================================================
Example complete! Check csg_v4_basic_example.png
==============================================================
```

Happy analyzing!
