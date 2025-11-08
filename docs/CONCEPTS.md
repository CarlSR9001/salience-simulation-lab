# CSG V4 Concepts: Salience Theory and Mathematics

This document provides a deep dive into the theoretical foundations of Continuity-Strain Gravity (CSG) Version 4, explaining the salience concept, mathematical formulas, and physical interpretations.

## Table of Contents

1. [Core Hypothesis](#core-hypothesis)
2. [The Salience Formula](#the-salience-formula)
3. [Scoring Components](#scoring-components)
4. [Quality Factor Transformation](#quality-factor-transformation)
5. [Interpolating Function](#interpolating-function)
6. [Physical Interpretation](#physical-interpretation)
7. [Workhorse Galaxies](#workhorse-galaxies)
8. [Parameter Tuning](#parameter-tuning)

---

## Core Hypothesis

**Salience-based gravity** proposes that gravitational response is not universal but depends on the "salience" of baryonic structure. Salience is a dimensionless measure combining:

- **Continuity**: How ordered vs. chaotic the motion is
- **Coherence**: What fraction of the system is ordered
- **Retention**: How stable the structure is over time
- **Mass**: How much baryonic matter is present
- **Morphology**: What type of structure it is

Regions with **high salience** exhibit stronger "inertial" or "gravitational" response, allowing visible matter to produce flat rotation curves without requiring dark matter.

### Analogy to MOND

MOND (Modified Newtonian Dynamics) proposes that gravity behaves differently at low accelerations. CSG extends this idea by making the transition acceleration **salience-dependent**:

- MOND: Single global a₀ for all systems
- CSG V4: Effective a_eff = κ_c × (a₀ / Q), where Q varies by location and galaxy type

This allows CSG to naturally explain:
- Why dwarf galaxies show stronger MOND-like effects (high gas fractions → high salience)
- Why grand design spirals have predictable rotation curves (high coherence → high salience)
- Why irregular galaxies are harder to model (low coherence → low salience)

---

## The Salience Formula

Salience S' is computed as a product of three terms:

```
S' = CORE × HYPE × PENALTY
```

### CORE: Fundamental Properties

```
CORE = C_local × R_galaxy × K_galaxy × M_local
```

This captures the "baseline salience" from basic galaxy properties:

- **C_local**: How continuous (ordered) is the rotation at each radius?
- **R_galaxy**: How well has the galaxy retained its structure over time?
- **K_galaxy**: What fraction of the disk is coherently rotating?
- **M_local**: How much baryonic mass is present at each radius?

All four factors must be present for high salience. If any is zero, CORE = 0.

### HYPE: Amplification

```
HYPE = 1 + (boost terms) - (penalty terms)

boost terms:
  + α_n × DeltaA_local      (morphology)
  + γ_X × X_galaxy          (edge case boost)
  + γ_Sh × S_h_galaxy       (stress-history)

penalty terms:
  - γ_W × W_local           (curvature irregularity)

HYPE ≥ 0.1  (floored for stability)
```

HYPE amplifies or reduces salience based on additional structural features:

- **DeltaA**: Morphology-dependent boost (higher for grand design spirals)
- **X**: Edge case boost for low-mass gas-rich coherent systems
- **S_h**: Stress-history boost (age + gas + dispersion)
- **W**: Curvature penalty (bumps/wiggles reduce salience)

HYPE is typically in the range [0.5, 2.0].

### PENALTY: Disorder Reduction

```
PENALTY = 1 - β_phi × phi_galaxy - β_J × J_galaxy

PENALTY ∈ [0, 1]
```

PENALTY reduces salience for structurally disordered systems:

- **phi**: Intrinsic disorder (higher for dwarfs/irregulars)
- **J**: Coherence gap (1 - max coherence)

Systems with high disorder or poor maximum coherence receive reduced salience.

### Final Salience

```
S' = CORE × HYPE × PENALTY
S' ≥ min_s_prime = 1e-10  (numerical floor)
```

S' is dimensionless and typically ranges from ~0.01 (low salience) to ~10 (high salience).

---

## Scoring Components

Each scoring component is carefully designed to capture a specific aspect of galaxy structure. All scores are normalized to [0, 1] unless otherwise noted.

### C_local: Local Continuity

**Formula:**
```
C_local(r) = 1 / (1 + σ(r) / V₀)

where:
  σ(r) = velocity dispersion at radius r [km/s]
  V₀ = reference velocity ≈ 50 km/s
```

**Physical Meaning:**
- Measures the ratio of ordered rotation (v) to random motion (σ)
- High C: Disk is coherently rotating, low turbulence
- Low C: High dispersion, turbulent, or measurement uncertainty

**Typical Values:**
- Grand design spiral inner disk: C ~ 0.7-0.9
- Dwarf galaxy: C ~ 0.5-0.7
- Irregular galaxy: C ~ 0.3-0.5

### R_galaxy: Retention

**Formula:**
```
R_galaxy = 0.5 × tanh(age / 5 Gyr) + 0.5 × median(v / (v + σ))
```

**Physical Meaning:**
- Age term: Older galaxies have had more time to settle into stable configurations
- Kinematic term: Systems dominated by rotation (v >> σ) are more stable
- Equal weighting of long-term (age) and current (kinematics) retention

**Typical Values:**
- Old, stable spiral: R ~ 0.8-0.9
- Young dwarf: R ~ 0.5-0.7

### K_galaxy: Coherence Fraction

**Formula:**
```
K_galaxy = (# of radii with C ≥ 0.6) / (total # of radii)
```

**Physical Meaning:**
- What percentage of the disk is "highly ordered"?
- Threshold C ≥ 0.6 corresponds to σ < V₀
- Measures global coherence, not just local peaks

**Typical Values:**
- Well-behaved spiral: K ~ 0.7-1.0
- Patchy spiral: K ~ 0.4-0.7
- Irregular: K ~ 0.2-0.4

### J_galaxy: Coherence Gap

**Formula:**
```
J_galaxy = 1 - max(C_local)
```

**Physical Meaning:**
- How far is the "best" region from perfect coherence?
- Even if average coherence is decent, a large J means no region is truly ordered
- Inverse measure: higher J is worse

**Typical Values:**
- Excellent coherence: J < 0.2 (max C > 0.8)
- Poor coherence: J > 0.5 (max C < 0.5)

### M_local: Mass Density Score

**Formula:**
```
Σ_baryon(r) = M_baryon(r) / (π r²)
M_local(r) = clip((log₁₀(Σ_baryon) - 6) / 4, 0, 1)
```

**Physical Meaning:**
- Maps log surface density from typical range [10⁶, 10¹⁰] M_sun/kpc² to [0, 1]
- Higher mass → stronger gravitational "anchor" → higher salience
- Normalized to account for exponential range of galaxy masses

**Typical Values:**
- Inner spiral disk: M ~ 0.6-0.9
- Outer disk: M ~ 0.2-0.5
- Dwarf galaxy: M ~ 0.3-0.6

### W_local: Curvature Irregularity

**Formula:**
```
dv/dr = gradient(v, r)
d²v/dr² = gradient(dv/dr, r)
curvature = |d²v/dr²|
W_local = clip(curvature / (max(curvature) × scale), 0, 1)

where scale ≈ 0.25
```

**Physical Meaning:**
- Measures "bumps" and "wiggles" in the rotation curve
- Second derivative picks up irregularities missed by first derivative
- High W indicates non-smooth structure → reduced salience

**Typical Values:**
- Smooth rotation curve: W < 0.3 everywhere
- Minor bumps: W ~ 0.3-0.6 in localized regions
- Highly irregular: W > 0.6 in many regions

### phi_galaxy: Structural Disorder

**Formula:**
```
phi_galaxy = morphology-dependent base value
  - Grand design / HSB spiral: 0.4
  - LSB dwarf / irregular: 0.6
  - Other: 0.5

Adjustments:
  - Unfair edge case: -0.2
  - Workhorse galaxy: -0.15
```

**Physical Meaning:**
- Intrinsic disorder based on morphological classification
- Lower phi means less penalty (better structure)
- "Unfair edge" adjustment for coherent low-mass systems

**Typical Values:**
- Well-organized: phi ~ 0.3-0.4
- Average: phi ~ 0.5
- Disordered: phi ~ 0.6-0.7

### X_galaxy: Edge Case Boost

**Formula:**
```
X_galaxy = 1 if (v_max < 70 km/s AND gas_fraction > 0.7 AND coherent_rotation)
           0 otherwise
```

**Physical Meaning:**
- Identifies "unfair edge" galaxies: low-mass, gas-rich, but coherently rotating
- These systems exhibit strong MOND-like effects despite low mass
- X boost compensates for phi penalty that would otherwise apply

**Typical Triggers:**
- DDO 154-like dwarfs
- Low surface brightness galaxies with organized rotation

### S_h_galaxy: Stress-History

**Formula:**
```
S_h = 0.4 × tanh(age / 5 Gyr)
    + 0.3 × gas_fraction
    + 0.3 × clip(median(σ/v) / 0.5, 0, 1)
```

**Physical Meaning:**
- Combines age, gas content, and stress (dispersion) into a "history" score
- Systems that have retained gas and experienced stress may have built up salience
- Weighted combination of three independent indicators

**Typical Values:**
- Old, gas-poor, low-stress: S_h ~ 0.4
- Young, gas-rich, high-stress: S_h ~ 0.7-0.9

### DeltaA_local: Morphology Boost

**Formula:**
```
base, r_feature = morphology-dependent constants
  - Grand design: (0.7, 5 kpc)
  - Spiral: (0.5, 5 kpc)
  - Dwarf: (0.3, 2 kpc)

radial_factor(r) = exp(-((r - r_feature) / r_feature)²)
DeltaA(r) = base × (0.5 + 0.5 × radial_factor)
```

**Physical Meaning:**
- Radial boost profile with Gaussian peak at characteristic radius
- Grand design spirals receive stronger boost than dwarfs
- Peak location varies: spirals peak at ~5 kpc, dwarfs at ~2 kpc
- Captures the idea that different morphologies have different "sweet spots"

**Typical Values:**
- Grand design at r_feature: DeltaA ~ 0.7
- Dwarf at r_feature: DeltaA ~ 0.3

---

## Quality Factor Transformation

Raw salience S' is transformed to a quality factor Q through two steps:

### Step 1: Local Quality Factor

```
Q_local(r) = [S'(r)]^(1/4)
```

**Why fourth root?**
- S' can vary over several orders of magnitude
- Direct use would cause numerical instability in division (a_eff = a₀ / Q)
- Fourth root compresses the range: S' ∈ [0.01, 10] → Q ∈ [0.3, 1.8]
- Still preserves ordering: higher S' → higher Q

### Step 2: Aura Blending

```
A_galaxy = mean(Q_local in inner 30% of radii)
Q_final(r) = (1 - α) × Q_local(r) + α × A_galaxy

where α = aura_mix ≈ 0.5
```

**Why blend local and global?**
- Pure local Q can fluctuate due to measurement noise
- Galaxy's "global character" (inner region average) provides stability
- Blend allows outer regions to "feel" the galaxy's overall quality
- α = 0.5: Equal weighting of local structure and global identity

**Effect:**
- Inner regions: Q_final ≈ Q_local (local dominates)
- Outer regions: Q_final pulled toward A_galaxy (global influences)

---

## Interpolating Function

The CSG V4 interpolating function smoothly transitions between Newtonian and MOND regimes:

### Effective Acceleration Scale

```
a_eff(r) = κ_c × (a₀ / Q_final(r))

where:
  κ_c = global calibration constant (fitted to data)
  a₀ = c × H₀ / 1000 ≈ 2.1e4 kpc/Gyr² (MOND scale)
  Q_final(r) = quality factor (typically 0.5-1.5)
```

**Physical Meaning:**
- a_eff sets the "transition acceleration" where MOND effects kick in
- Higher Q (higher salience) → lower a_eff → more MOND-like at lower accelerations
- κ_c allows global calibration without changing the salience structure

**Typical Values:**
- Grand design spiral (Q ~ 1.2): a_eff ~ 1.8e4 kpc/Gyr² (with κ_c ~ 1)
- Dwarf galaxy (Q ~ 0.8): a_eff ~ 2.6e4 kpc/Gyr²

### Interpolation Weight

```
x(r) = g_bar(r) / a_eff(r)
μ(r) = x(r) / (1 + x(r))
```

**Physical Meaning:**
- x: Dimensionless ratio of actual to transition acceleration
- μ: Smooth interpolation weight ∈ [0, 1]
- μ → 1 as x → ∞ (high acceleration, Newtonian regime)
- μ → 0 as x → 0 (low acceleration, MOND regime)

### Predicted Acceleration

```
g_obs(r) = μ(r) × g_bar(r) + (1 - μ(r)) × √[a_eff(r) × g_bar(r)]
```

**Physical Meaning:**
- Linear blend of two limiting behaviors:
  1. **Newtonian**: g_obs = g_bar (when μ = 1)
  2. **Deep MOND**: g_obs = √(a_eff × g_bar) (when μ = 0)
- Smooth transition region where both contribute

### Predicted Velocity

```
v_pred(r) = √[g_obs(r) × r]
```

Standard circular velocity from centripetal acceleration.

### Asymptotic Regimes

**Inner regions (g_bar >> a_eff):**
```
x >> 1 → μ ≈ 1
g_obs ≈ g_bar
v_pred ≈ √(g_bar × r)  [Newtonian]
```

**Outer regions (g_bar << a_eff):**
```
x << 1 → μ ≈ x = g_bar / a_eff
g_obs ≈ √(a_eff × g_bar)
v_pred ≈ (a_eff × r)^(1/4) × g_bar^(1/4)  [MOND-like]
```

In the MOND regime, v_pred depends on g_bar^(1/4), producing nearly flat rotation curves as g_bar slowly declines with radius.

---

## Physical Interpretation

### Why Does Salience Matter?

CSG V4 proposes that gravitational response is **not universal** but depends on the **informational coherence** of baryonic structure:

1. **Coherent systems** (high C, high K):
   - Velocity vectors "agree" with each other
   - System behaves as a unified entity
   - Gravitational response is "efficient"
   - → Lower Q, lower a_eff, more MOND-like

2. **Incoherent systems** (low C, low K):
   - Velocity vectors are random or conflicting
   - System is fragmented or chaotic
   - Gravitational response is "diluted"
   - → Higher Q, higher a_eff, more Newtonian

This is analogous to how:
- A coherent laser beam carries more "effective" energy than incoherent light
- A synchronized crowd has more "collective strength" than individuals
- An organized army is more effective than a mob

### Connection to Continuity and Strain

The name "Continuity-Strain Gravity" reflects two key ideas:

1. **Continuity**: Systems with continuous, smooth, coherent structure have higher salience
2. **Strain**: Discontinuities, curvature, disorder create "strain" that reduces salience

The CORE term captures baseline continuity. The HYPE and PENALTY terms capture strain and disorder.

### Why Fourth-Root Scaling?

The Q = S'^(1/4) transformation serves multiple purposes:

- **Numerical stability**: Prevents division by tiny or huge numbers in a_eff = a₀ / Q
- **Physical motivation**: In MOND, v ~ g_bar^(1/4) in the deep regime. The fourth root in Q produces this naturally.
- **Compression**: Maps wide salience range to moderate Q range

### The Role of κ_c

κ_c is the global calibration constant that absorbs:
- Uncertainties in absolute salience scale
- Normalization conventions for Q
- Overall gravitational coupling strength

It is fitted to minimize residuals across a sample of galaxies. Typical value: κ_c ~ 0.3-0.8 depending on parameter choices.

---

## Workhorse Galaxies

"Workhorse" galaxies are high-quality spirals that meet strict criteria:

```
Workhorse if:
  - Type in {hsb_spiral, grand_design, spiral}
  - R_galaxy ≥ 0.85  (high retention)
  - K_galaxy ≥ 0.75  (high coherence)
  - mean(C_local) ≥ 0.7  (high average continuity)
  - mean(M_local) ≥ 0.3  (sufficient mass)
```

### Special Treatment

Workhorse galaxies receive bonuses:

1. **Reduced disorder penalty**: phi → phi - 0.15
2. **Reduced curvature penalty**: W → W × 0.7
3. **HYPE bonus**: +0.35 to HYPE term

### Physical Justification

Workhorse galaxies are the "gold standard" systems:
- Reliable measurements (high coherence → low scatter)
- Stable structure (high retention → predictable behavior)
- Sufficient mass (M ≥ 0.3 → gravity dominates over noise)

They should not be penalized for minor irregularities that stem from measurement noise rather than true disorder. The workhorse bonuses acknowledge this.

---

## Parameter Tuning

### Key Parameters and Their Effects

| Parameter | Default | Effect if Increased |
|-----------|---------|---------------------|
| `alpha_n` | 0.35 | Stronger morphology boost (DeltaA) |
| `gamma_X` | 0.25 | Stronger edge case boost |
| `gamma_Sh` | 0.30 | Stronger stress-history boost |
| `gamma_W` | 0.55 | Stronger curvature penalty |
| `beta_phi` | 0.60 | Stronger disorder penalty |
| `beta_J` | 0.45 | Stronger coherence gap penalty |
| `gamma_workhorse` | 0.35 | Larger workhorse bonus |
| `aura_mix` | 0.5 | More global influence on Q_final |

### Optimization Strategy

1. **Fix physical constants**: G, c, H0, V0 from observations
2. **Fix thresholds**: Workhorse criteria, coherence threshold (C ≥ 0.6)
3. **Scan κ_c**: Grid search over [0.001, 1.0] to find optimal fit
4. **Refine**: Quadratic fit or golden section search around best κ_c
5. **Uncertainty**: Bootstrap resampling for confidence intervals

### Validation

- **Residual plots**: Check for systematic trends vs. radius, velocity, galaxy type
- **Parameter stability**: Small changes in config should not drastically change κ_c
- **Physical plausibility**: Scores should correlate with intuition (e.g., grand design → high K)

---

## Summary

CSG V4 is built on the hypothesis that **gravitational response depends on salience**, a composite measure of continuity, coherence, retention, mass, and morphology. The model:

1. Computes salience S' = CORE × HYPE × PENALTY
2. Transforms to quality factor Q = S'^(1/4), blended with global aura
3. Uses Q to set effective acceleration scale a_eff = κ_c × (a₀ / Q)
4. Interpolates between Newtonian and MOND regimes via μ(x) = x/(1+x)
5. Predicts flat rotation curves matching observations without dark matter

The framework is **morphology-aware**, **coherence-sensitive**, and **mass-dependent**, allowing it to naturally explain diversity in galaxy dynamics.

For implementation details, see the source code in `src/csg_v4/`.
For usage examples, see `docs/examples/` and `README.md`.

---

**Questions?** Open an issue or refer to `AGENTS.md` for experimental context.
