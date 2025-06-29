# PH550-Supplementary Material
PH550 Project Repost Supplementary Material
## KerrDTS.py and BECDTS.py are the scrpts used to simulate the models described by the PRL report.
The scripts were majority autored by me (Zhanming Mei), and partially based off the scripts provided by supervisor G.W.Henderson.


--------------------------------------------------------------------------------------------
## Adaptive Step size utilisation

RFK45 adaptive time step can be enabled by changed the toggle in BECDTS.py, and setting to "False" reverts the code back
to fixed step size with relaxation method as described by the Model handbook by G.W.Henderson.

--------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------
## Exel Tables

There are 4 Excel tables, 3 for paramters sweep and 1 with the parameters and their corresponding patterns.
Table names are self explanatory.

--------------------------------------------------------------------------------------------


## Quick Guide
--------------------------------------------------------------------------------------------
### Numeriocal Stability and Otical Saturation in BEC Cavity Model 

You may strugle with
numerical stability with BEC Cavity simulations, be  ware of the Sigma value (optical saturatuion), and try out different once until 
numerical stability is reached where the different interactions are balanced and stable patterns will arise/anneal with a high enough precision 
simulation run. Highly recommend running all simulations with 512x512 grid for suffcient stability and resolution.

--------------------------------------------------------------------------------------------


## Solver Parameters Fine Tuning
--------------------------------------------------------------------------------------------
### Relaxation Method 

For Relaxation parameters, I'ts recommended to set tolerance to smaller than 1e-5 but using 1e-4 to obtain a rough picture of the pattern on set before commting to run the long 
simulaiton with higher numerical precision.
### Adaptive Step Size RKF45

The adaptive step size moethod can be very tricky to "get right" since if the field's evolved does not converge properly, the simulation could get stuck repeat in a loop.
the recommended tolerance here is again between 1e-4 to 1e-7 for the range of step size allowed for a high precision run while balancing simulation time and numerical stability

--------------------------------------------------------------------------------------------


## Error checks/Convergence test
--------------------------------------------------------------------------------------------
The simulation utilises a sophisticated, two-pronged approach to ensure numerical accuracy and stability. It independently manages the convergence of the fast-evolving optical field and controls the truncation error of the slow-evolving atomic field's dynamics.

#### 1. Optical Field Steady-State Convergence

To handle the rapid evolution of the intra-cavity optical field, the simulation employs a Relaxation Method. Instead of co-evolving the optical field in time, it is calculated at each atomic time step by evolving it to its steady state.

The convergence criterion for this steady state is defined within the relax_optical_field_to_steady_state function:

The simulation tracks the mean amplitude of the optical field over its most recent evolution steps.

It then calculates the standard deviation of these tracked values. A steady state is considered to have been reached when this standard deviation becomes negligibly small relative to the mean amplitude, falling below a user-defined tolerance.

This ensures that the optical field used to influence the atoms is stable and is not undergoing transient oscillations.

#### 2. Atomic Field Truncation Error Control (RKF45)

The primary mechanism for controlling the accuracy of the main simulation is the Runge-Kutta-Fehlberg 45 (RKF45) method. This is an adaptive step-size technique that guarantees the numerical error remains within a pre-defined limit at each step of the atomic field's evolution.

The process is as follows:

At each step in time, the simulation calculates two solutions for the atomic field: one with a fourth-order Runge-Kutta method (RKF4) and another with a fifth-order one (RKF5).

The absolute difference between these two solutions provides a robust estimate of the local truncation error, ε_T.

This error is compared against a maximum permitted truncation error, ε_Tmax, which is set by the user (rkf_tolerance).

If the calculated error ε_T is greater than the allowed tolerance ε_Tmax, the step is rejected. A new, smaller time step is then calculated, and the step is attempted again.

If the error is within the tolerance, the more accurate fifth-order solution is accepted, and the simulation proceeds. An optimal step size for the next iteration is also calculated based on the most recent error estimation.

This dynamic process ensures that the simulation automatically takes smaller steps during periods of complex evolution and larger steps when the dynamics are simpler, maintaining a consistent level of accuracy whilst maximising efficiency.

#### 3. Numerical Instability Check

As a final safeguard, a basic check is performed during the evolution loop to detect numerical breakdown. If the simulation values become non-physical (e.g., resulting in a NaN, or 'Not a Number' value), typically due to extreme nonlinearities, the evolution is terminated to prevent generating erroneous results.

--------------------------------------------------------------------------------------------


## Data Analysis/Interpretation
--------------------------------------------------------------------------------------------
The script AnalysisTool.py is for the output data from the Cavity simulations (BECDTS.py/KerrDTS.py).
It takes the CSV files representing the complex atomic field (Psi) and performs
two key analyses:

#### 1.  Lattice Constant (Λ) Calculation:
Computes the 2D Fast Fourier Transform (FFT) of the atomic intensity |Psi|**2.

Identifies the primary peaks in the Fourier spectrum, which correspond to the
fundamental spatial frequencies of the pattern.

Calculates the lattice constant Λ = 2*pi/|k|, where |k| is the wavevector
magnitude of the pattern.

Estimates uncertainty from the standard deviation of the identified peak distances.

#### 2.  Rotational Order Parameter (Psi_4 and Psi_6) Calculation:
Finds local maxima (i.e., the bright spots) in the real-space intensity pattern.

Calculates the four-fold (Psi_4) and six-fold (Psi_6) rotational order
parameters to quantitatively measure the pattern's symmetry.

Classifies the pattern as hexagonal, square, or disordered based on
the calculated order parameter values.

--------------------------------------------------------------------------------------------

