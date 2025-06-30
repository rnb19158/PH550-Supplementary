# Supplementary Material for "Morphology-Dependent Pattern Selection in a Cavity-Driven Bose-Einstein Condensate"

Zhanming Mei, PH550 Project Report, University of Strathclyde (2025)

---

## Code Availability

### Main Simulation Scripts
- `KerrDTS.py`: Simulation of pattern formation in Kerr medium (baseline comparison)
- `BECDTS.py`: BEC-cavity system simulation implementing coupled LLE-GPE dynamics
- `AnalysisTool.py`: Pattern analysis including lattice constant extraction and order parameter calculation

*Note: Scripts were primarily authored by Z. Mei, with initial framework partially based on code provided by supervisor G.W. Henderson.*

---

## Data Files

### Parameter Tables
Four Excel tables are provided containing systematic parameter sweeps:
- `Param Sweep 1.xlsx`: Cavity detuning variation (θ = 0 to 4)
- `Param Sweep 2.xlsx`: parameter space exploration
- `Param Sweep 2.xlsx`: wider parameter space exploration with Collisional parameter variation (βcol = -2.5 to 2.5)
- `Pattern-parameters.xlsx`: Complete mapping of parameters to observed patterns

---

## Numerical Methods and Convergence

### Adaptive Step Size Implementation
The RKF45 adaptive timestepping can be enabled/disabled via the toggle in `BECDTS.py`. Setting to `False` reverts to fixed-step evolution with the relaxation method as described in the theoretical framework.

### Convergence Criteria

#### 1. Optical Field Steady-State Convergence
The simulation employs a relaxation method for the rapidly-evolving optical field. At each atomic timestep, the optical field is evolved to steady state using:
- Convergence tolerance: 10^-6 (relative to mean field amplitude)
- Recommended for initial testing: 10^-4 (faster but lower precision)
- Production runs: 10^-5 to 10^-6

#### 2. Atomic Field Evolution (RKF45)
Adaptive step-size control maintains local truncation error within specified bounds:
- Recommended tolerance: 10^-5 to 10^-8
- Fourth and fifth-order solutions compared at each step
- Step rejected and reduced if error exceeds tolerance
- Optimal balance between accuracy and computational efficiency

#### 3. Numerical Stability Safeguards
- Automatic termination upon NaN detection
- Monitoring of optical saturation parameter (σ)
- Grid resolution: 512×512 recommended for stability

---

## Practical Usage Guide

### Quick Start
For stable pattern formation in BEC-cavity simulations:
1. Monitor the optical saturation parameter (σ) - adjust if instabilities occur
2. Use 512×512 spatial grid for sufficient resolution
3. Start with relaxation tolerance 10^-4 for parameter exploration
4. Refine to 10^-6 for publication-quality results

### Common Issues and Solutions
- **Numerical instabilities**: Adjust σ value to balance competing interactions
- **Slow convergence**: Check if parameters lie near phase boundaries
- **RKF45 getting stuck**: Occurs near critical points - try tighter tolerance or switch to fixed-step

---

## Data Analysis Methods

### Lattice Constant Determination
The `AnalysisTool.py` script processes simulation output (CSV files) to extract:

1. **Fourier Analysis**:
   - 2D FFT of atomic density |ψ|²
   - Peak identification in k-space
   - Lattice constant Λ = 2π/|k|
   - Uncertainty from standard deviation of peak positions

2. **Order Parameter Calculation**:
   - Local maxima detection (intensity threshold: 0.5 × max(I))
   - Six-fold order: Ψ₆ = |Σⱼ exp(6iφⱼ)|/N
   - Four-fold order: Ψ₄ = |Σⱼ exp(4iφⱼ)|/N
   - Pattern classification: hexagonal (Ψ₆ > 0.8), square (Ψ₄ > 0.8), or disordered

### Phase Boundary Identification
Phase transitions determined when:
- Order parameter discontinuity: ΔΨ > 0.5
- Symmetry crossover: |Ψ₆ - Ψ₄| < 0.1
- Uncertainty in boundary position: ±0.1 in θ, ±0.05 in βcol

---

## Reproducibility

To reproduce key results from the paper:
1. Figure 3 (BEC pattern): θ = 1.75, βcol = 1.5, σ = appropriate value for stability
2. Figure 5 (hexagonal lattice): θ = 1.5, βcol = 1.5, top-hat pump with rₚ = 10w, σ = appropriate value for stability
3. Phase diagram: Run parameter sweeps using provided Excel templates


---

## Version Information
- Python 3.8+
- NumPy 1.19+
- SciPy 1.5+
- Matplotlib 3.3+

Last updated: June 2025
