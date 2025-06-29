#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:09:03 2025

@author: zhanmingmei

BEC-Cavity Coupled System Simulator - Adaptive Step-Size Version

This script simulates the coupled dynamics of a Bose-Einstein Condensate (BEC)
and an optical field within a driven cavity. It solves the Gross-Pitaevskii
Equation (GPE) for the atoms and finds the steady-state of the Lugiato-Lefever
Equation (LLE) for the light at each atomic time step.

This version implements the "Relaxation Method" described in Section 12.2
of the Model Handbook. The fast optical dynamics are evolved to steady-state
for a given (static) atomic field, and this steady-state optical field is
then used to evolve the atomic field over a single slow time step.

(v16 - Adaptive Step Size):
- Implemented an adaptive Runge-Kutta-Fehlberg 45 (RKF45) method for the
  atomic field evolution, as described in Section 14 of the Model Handbook.
- Added a 'use_adaptive_step' flag to the user controls to switch between
  the new adaptive method and the original fixed-step RK2 method.
- Added parameters for the adaptive step controller: 'rkf_tolerance',
  'd_tau_s_initial', 'd_tau_s_min', and 'd_tau_s_max'.
- The main evolution loop has been updated to handle step rejection and
  dynamic recalculation of the step size `d_tau_s`.
- A new function, `rkf45_step_psi`, has been added to perform the RKF45
  calculation for the atomic field's nonlinear evolution.

VERIFICATION: The implementation has been cross-referenced with the
'Model_Handbook.pdf' (Sec. 12.2, 14, Eqs. 56, 57 & 58) to confirm the implementation of the physical model and
numerical methods.
"""

# --- Core Libraries ---
import numpy as np
import os
import sys
from datetime import datetime
from collections import deque

# --- Scientific & Plotting Libraries ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend to prevent GUI windows
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
import scipy.constants as const


###############################################################################
# SECTION 1: PHYSICAL CONSTANTS AND UNIT CONVERSIONS
# This section defines fundamental physical constants for calculations.
###############################################################################

# Fundamental constants from scipy
c_light = const.c               # Speed of light [m/s]
hbar = const.hbar               # Reduced Planck constant [J·s]
epsilon_0 = const.epsilon_0     # Vacuum permittivity [F/m]
a_0 = const.physical_constants['Bohr radius'][0]  # Bohr radius [m]

# Cesium-133 atomic parameters (example)
m_Cs = 132.90545196 * const.atomic_mass # Cesium mass [kg]
lambda_Cs_D2 = 852.34727582e-9      # D2 line wavelength [m]
d_Cs = 2.989e-29                    # D2 transition dipole moment [C·m]

###############################################################################
# SECTION 2: CORE CALCULATION AND UTILITY FUNCTIONS
# These functions perform the core physics calculations and generate initial fields.
###############################################################################

def calculate_dimensionless_parameters(p):
    """Calculates dimensionless simulation parameters from physical inputs."""
    w_s = p['w_s_um'] * 1e-6
    delta_omega_cavity = 2 * np.pi * p['cavity_detuning_MHz'] * 1e6
    Delta_atom_light = 2 * np.pi * p['atom_detuning_MHz'] * 1e6
    a_s = p['scattering_length_a0'] * a_0
    k_L_pump = 2 * np.pi / p['wavelength_m']

    theta_mh = -2 * delta_omega_cavity * p['cavity_length_m'] / (c_light * p['T_mirror'])
    alpha_F = (2 * p['medium_length_m']) / (k_L_pump * w_s**2 * p['T_mirror'])
    beta_F = (2 * p['medium_length_m']) / (k_L_pump * w_s**2 * p['T_mirror'])
    
    if abs(Delta_atom_light) < 1e-9:
        print("Warning: Atom-light detuning is zero. beta_col might be ill-defined.")
        beta_col = np.inf
    else:
        mu_squared = p['dipole_moment_Cm']**2
        beta_col = (16 * np.pi * epsilon_0 * hbar * a_s * np.abs(Delta_atom_light)) / (k_L_pump**2 * mu_squared)
    
    s = -1 if Delta_atom_light < 0 else 1
    
    return {
        'theta': theta_mh, 'alpha_F': alpha_F, 'beta_F': beta_F,
        'beta_col': beta_col, 's': s,
    }

def calculate_psi_derivative(Psi, A_steady, nl_params):
    """Calculates the nonlinear derivative for the atomic field (GPE), as per Handbook Eq. 57."""
    s, beta_dd, beta_col, L_3 = nl_params['s'], nl_params['beta_dd'], nl_params['beta_col'], nl_params.get('L_3', 0.0)
    abs_Psi_sq = np.abs(Psi)**2
    abs_A_sq = np.abs(A_steady)**2
    
    nl_Psi_terms = (-1j * s * abs_A_sq + 1j * 2.0 * beta_dd * abs_A_sq * abs_Psi_sq - 1j * beta_col * abs_Psi_sq - L_3 * abs_Psi_sq**2)
    dPsidt_nl = nl_Psi_terms * Psi
    
    return dPsidt_nl

def calculate_A_derivative(A, Psi_static, F_P, nl_params):
    """Calculates the nonlinear derivative for the optical field (LLE) for a static Psi, as per Handbook Eq. 56."""
    beta_F, s, beta_dd, sigma = nl_params['beta_F'], nl_params['s'], nl_params['beta_dd'], nl_params.get('sigma', 0.0)
    abs_Psi_sq = np.abs(Psi_static)**2
    abs_A_sq = np.abs(A)**2

    nl_A_coupling = -1j * beta_F * (s * abs_Psi_sq - beta_dd * abs_Psi_sq**2) / (1.0 + sigma * abs_A_sq + 1e-12)
    dAdt_nl = (nl_A_coupling * A) + F_P
    
    return dAdt_nl

def relax_optical_field_to_steady_state(Psi_static, A_initial, F_P, nl_params, lin_op_a, relax_params):
    """
    Evolves the optical field 'A' to its steady state for a given static atomic field 'Psi'.
    This function implements the core idea of the relaxation method from Sec 12.2 of the handbook.
    """
    d_tau_fast = relax_params['d_tau_fast']
    max_steps = relax_params['max_steps']
    tolerance = relax_params['tolerance']

    A_current = A_initial.copy()
    A_fft = np.fft.fft2(A_current)
    convergence_tracker = deque(maxlen=20) 

    for step in range(max_steps):
        A_fft *= lin_op_a
        A_curr_nl = np.fft.ifft2(A_fft)
        
        dAdt = calculate_A_derivative(A_curr_nl, Psi_static, F_P, nl_params)
        A_next = A_curr_nl + d_tau_fast * dAdt
        A_fft = np.fft.fft2(A_next)
        
        A_fft *= lin_op_a

        if step % 10 == 0:
            A_final_step = np.fft.ifft2(A_fft)
            mean_amp = np.mean(np.abs(A_final_step))
            convergence_tracker.append(mean_amp)

            if len(convergence_tracker) == convergence_tracker.maxlen:
                conv_val = np.std(convergence_tracker) / (mean_amp + 1e-9)
                if conv_val < tolerance:
                    return A_final_step
                
                if step > 0 and step % 200 == 0:
                    print(f"    ... relaxation step {step}, convergence: {conv_val:.2e} (target: {tolerance:.2e})")

    return np.fft.ifft2(A_fft)

def rkf45_step_psi(Psi_0, A_0, d_tau_s, F_P, nl_params, lin_op_a_relax, relax_params):
    """
    Performs a single adaptive RKF45 step for the atomic field's nonlinear evolution.
    This function is based on the method described in Sec. 14 of the Model Handbook.
    """
    # Butcher Tableau coefficients for RKF45
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    a = np.array([
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ])
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    k = [0]*6
    A_steady = [A_0] * 7 # Store steady-state fields for each stage

    # Stage 1
    A_steady[0] = relax_optical_field_to_steady_state(Psi_0, A_0, F_P, nl_params, lin_op_a_relax, relax_params)
    k[0] = calculate_psi_derivative(Psi_0, A_steady[0], nl_params)
    
    # Stages 2-6
    for i in range(1, 6):
        Psi_temp = Psi_0 + d_tau_s * sum(a[i, j] * k[j] for j in range(i))
        # Use previous stage's A_steady as initial guess for faster relaxation
        A_steady[i] = relax_optical_field_to_steady_state(Psi_temp, A_steady[i-1], F_P, nl_params, lin_op_a_relax, relax_params)
        k[i] = calculate_psi_derivative(Psi_temp, A_steady[i], nl_params)

    # Calculate 4th and 5th order solutions
    Psi_rk4 = Psi_0 + d_tau_s * sum(b4[i] * k[i] for i in range(6))
    Psi_rk5 = Psi_0 + d_tau_s * sum(b5[i] * k[i] for i in range(6))
    
    # Estimate truncation error (absolute, not relative)
    error = np.sum(np.abs(Psi_rk5 - Psi_rk4))
    
    # Return the more accurate 5th order solution, the final optical field, and the error
    return Psi_rk5, A_steady[5], error

def generate_lg_pump(X, Y, F_amp, w0_phys, l, l_prof, p, noise_factor):
    """
    Generates a Laguerre-Gaussian (LG) pump beam.
    Decouples phase from amplitude profile.
    l: Azimuthal index for the amplitude profile.
    l_prof: Azimuthal index for the phase profile (OAM).
    """
    r_sq = X**2 + Y**2; w0_phys_sq = w0_phys**2
    if w0_phys_sq < 1e-9: return np.zeros_like(X, dtype=np.complex64)
    
    radial_norm_sq = r_sq / w0_phys_sq
    phi = np.arctan2(Y, X)
    
    # Amplitude profile is determined by `l` and `p`
    laguerre_poly = assoc_laguerre(2 * radial_norm_sq, p, np.abs(l))
    amplitude_profile = (np.sqrt(radial_norm_sq))**np.abs(l) * laguerre_poly * np.exp(-radial_norm_sq)
    
    # Phase profile (OAM) is determined by `l_prof`
    phase_profile = np.exp(1j * l_prof * phi)
    
    lg_profile = amplitude_profile * phase_profile
    max_abs = np.max(np.abs(lg_profile))
    norm_lg = F_amp * (lg_profile / max_abs if max_abs > 1e-9 else lg_profile)
    
    return norm_lg.astype(np.complex64) + generate_noise(X.shape[1], X.shape[0], F_amp, noise_factor)

def generate_thomas_fermi_bec(X, Y, Psi_amp, tf_radius, noise_factor):
    """Generates a Thomas-Fermi initial BEC profile."""
    r_sq = X**2 + Y**2; tf_radius_sq = tf_radius**2
    if tf_radius_sq < 1e-9: profile = np.zeros_like(X, dtype=np.float32)
    else: profile = Psi_amp * np.sqrt(np.maximum(0, 1 - r_sq / tf_radius_sq))
    return profile.astype(np.complex64) + generate_noise(X.shape[1], X.shape[0], Psi_amp, noise_factor)

def generate_top_hat_pump(X, Y, F_amp, radius_um, steepness, l_prof, noise_factor):
    """Generates a top-hat pump profile with optional OAM."""
    r = np.sqrt(X**2 + Y**2)
    # Amplitude profile
    amplitude_profile = F_amp * (0.5 * (1 - np.tanh(steepness * (r - radius_um))))
    
    # Phase profile (OAM) is determined by `l_prof`
    phi = np.arctan2(Y, X)
    phase_profile = np.exp(1j * l_prof * phi)
    
    # Combine amplitude and phase
    profile = amplitude_profile * phase_profile
    
    return profile.astype(np.complex64) + generate_noise(X.shape[1], X.shape[0], F_amp, noise_factor)

def generate_homogeneous_field(X, Y, amp, noise_factor):
    """Generates a uniform (homogeneous) field."""
    profile = np.full(X.shape, amp, dtype=np.complex64)
    return profile + generate_noise(X.shape[1], X.shape[0], amp, noise_factor)

def generate_noise(Nx, Ny, amp, noise_amplitude, dtype=np.complex64):
    """Creates a field of random complex noise."""
    noise_re = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    noise_im = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    return (amp * noise_amplitude * (noise_re + 1j * noise_im)).astype(dtype)

def create_absorbing_boundaries(X_um, Y_um, x_um, Lx_um, w_s_um, boundary_extent, boundary_grad):
    """
    Creates smooth, radially-symmetric absorbing boundaries based on the FOAMilyCAC.py implementation.
    This prevents field wrapping at grid edges.
    """
    # N_w is the number of characteristic beam waists in the domain size.
    N_w = Lx_um / w_s_um
    
    # Position where the absorbing boundary starts, relative to the grid extent.
    boundary_lim = np.max(x_um) * boundary_extent
    
    # Calculate the boundary mask using tanh. This is the core logic from FOAMilyCAC.py.
    # The term (N_w / 30) is a scaling factor from the original library.
    boundaries = 1 - np.tanh((boundary_grad / (N_w / 30)) * (np.sqrt(X_um**2 + Y_um**2) - boundary_lim))
    
    # Re-scale the result to be strictly between 0 and 1.
    boundaries = np.interp(boundaries, (boundaries.min(), boundaries.max()), (0, 1))
    
    return boundaries.astype(np.float32)

###############################################################################
# SECTION 3: DATA SAVING AND PLOTTING
# These functions handle file I/O, saving results as plots and raw data.
###############################################################################

def setup_output_directories(base_dir, overwrite):
    """Ensures the base output directory and a 'csv_data' subdirectory exist."""
    if os.path.exists(base_dir) and overwrite == 0:
        print(f"Error: Output directory '{base_dir}' already exists and overwrite is disabled.")
        print("Set 'overwrite = 1' to proceed, or choose a new 'output_dir_name'.")
        sys.exit()
    os.makedirs(base_dir, exist_ok=True)
    csv_dir = os.path.join(base_dir, 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)
    return base_dir, csv_dir

def calculate_spectrum_log(field):
    """Calculates the spatial power spectrum in log scale for plotting."""
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    return np.log10(spec + 1e-12)

def save_all_plots(Psi, A, F_P, directory, base_filename, grid_params, track_to_max, tracked_maxs, plot_window_um):
    """Saves plots of all relevant fields, with optional max amplitude tracking and plot zoom."""
    x_um, y_um = grid_params['x'], grid_params['y']
    kx_um, ky_um = grid_params['kx'], grid_params['ky']
    extent_real = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]
    
    kx_shifted = np.fft.fftshift(kx_um); ky_shifted = np.fft.fftshift(ky_um)
    extent_k = [kx_shifted.min(), kx_shifted.max(), ky_shifted.min(), ky_shifted.max()]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.1) 

    fields = {'BEC (Psi)': Psi, 'Optical (A)': A, 'Pump (F_P)': F_P}
    cmaps = {'amp': 'viridis', 'phase': 'hsv', 'spec': 'magma'}

    for i, (name, field) in enumerate(fields.items()):
        amp = np.abs(field)
        phase = np.angle(field) + np.pi 
        spec = calculate_spectrum_log(field)

        vmax_val = None
        if track_to_max == 1:
            current_max = np.max(amp)
            if current_max > tracked_maxs[i]: tracked_maxs[i] = current_max
            vmax_val = tracked_maxs[i] if tracked_maxs[i] > 1e-9 else None
        
        ax_amp = axes[i, 0]
        im_amp = ax_amp.imshow(amp, origin='lower', extent=extent_real, cmap=cmaps['amp'], aspect='auto', vmax=vmax_val)
        ax_amp.axis('off')
        cbar_amp = fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
        
        max_amp_val = vmax_val if track_to_max == 1 and vmax_val is not None else np.max(amp)
        if max_amp_val > 1e-9:
             cbar_amp.set_ticks([np.min(amp), max_amp_val])
             cbar_amp.set_ticklabels(['min', f'{max_amp_val:.2f}'])
        else:
             cbar_amp.set_ticks([0]); cbar_amp.set_ticklabels(['0'])

        ax_phase = axes[i, 1]
        im_phase = ax_phase.imshow(phase, origin='lower', extent=extent_real, cmap=cmaps['phase'], vmin=0, vmax=2*np.pi, aspect='auto')
        ax_phase.axis('off')
        cbar_phase = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
        cbar_phase.set_ticks([0, 2 * np.pi]); cbar_phase.set_ticklabels(['0', '2π'])

        if plot_window_um > 0:
            plot_half_width = plot_window_um / 2
            ax_amp.set_xlim(-plot_half_width, plot_half_width)
            ax_amp.set_ylim(-plot_half_width, plot_half_width)
            ax_phase.set_xlim(-plot_half_width, plot_half_width)
            ax_phase.set_ylim(-plot_half_width, plot_half_width)

        ax_spec = axes[i, 2]
        im_spec = ax_spec.imshow(spec, origin='lower', extent=extent_k, cmap=cmaps['spec'], aspect='auto')
        ax_spec.axis('off')
        im_spec.set_clim(np.percentile(spec, 5), spec.max())
        fig.colorbar(im_spec, ax=ax_spec, fraction=0.046, pad=0.04)

    full_path = os.path.join(directory, f"{base_filename}.png")
    plt.savefig(full_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved plot: {full_path}")
    plt.close(fig)
    return tracked_maxs

def save_field_data_csv(field, directory, base_filename, field_name):
    """Saves the real and imaginary parts of a complex field to CSV."""
    real_path = os.path.join(directory, f"{base_filename}_{field_name}_real.csv")
    imag_path = os.path.join(directory, f"{base_filename}_{field_name}_imag.csv")
    np.savetxt(real_path, field.real, delimiter=','); np.savetxt(imag_path, field.imag, delimiter=',')
    print(f"Saved data for {field_name} to {directory}")

def save_simulation_info(output_dir, csv_dir, params_dict, finished=False):
    """Saves simulation parameters and timestamps to a log file."""
    info_file_path = os.path.join(output_dir, "simulation_info.txt")
    mode = 'a' if os.path.exists(info_file_path) else 'w'
    with open(info_file_path, mode) as f:
        if not finished:
            f.write(f"\nBEC-Cavity Simulation (START) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n")
            f.write(f"Plot output: {os.path.abspath(output_dir)}\nCSV data: {os.path.abspath(csv_dir)}\n\nParameters:\n")
            for key, value in params_dict.items(): f.write(f"  {key:<30} = {value}\n")
            f.write("\n")
        else:
            f.write(f"\nBEC-Cavity Simulation (END) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n")

###############################################################################
# SECTION 4: MAIN SIMULATION
# This is the main executable part of the script.
###############################################################################
def main():
    """Main function to configure and run the BEC-Cavity simulation."""
    
    #==========================================================================
    # --- USER CONTROL PANEL ---
    # This is the primary section controlling the simulation parameters.
    #==========================================================================

    # --- A) Simulation Scaling ---
    # w_s_um is the characteristic scaling waist. It defines the grid and dimensionless parameters.
    w_s_um = 20.0         # Characteristic SCALING beam waist in microns.
    
    # --- B) Physical & Parameter Style ---
    use_physical_params = False 
    wavelength_m = 720e-9  # Pump wavelength in meters (used for beta_dd calculation regardless)

    # --- C) Initial Field Profiles ---
    # Select the initial shape for the pump and BEC fields.
    # Options: "Homogeneous", "LG" (Laguerre-Gauss), "TopHat", "TF" (Thomas-Fermi for BEC)
    pump_profile = "Homogeneous" 
    bec_profile = "Homogeneous"

    # --- D) Dimensionless Physics Parameters (Used if use_physical_params is False) ---
    # These are the direct inputs to the simulation equations. Easiest for parameter sweeps.
    direct_nl_params = {
        'theta': 1.0,       # Cavity detuning
        'alpha_F': 1.0,     # Optical diffraction strength
        'beta_F': 1.0,      # Optical nonlinearity (atom-light coupling) strength
        's': -1.0,          # Sign of atom-light detuning (-1 for red, +1 for blue)
        'beta_col': 1.5,    # Atomic collisional (self-interaction) strength
        'L_3': 0.00022,     # Three-body loss coefficient
        'sigma': 0.0,       # Optical saturation
    }
    
    # --- E) Time Evolution & Relaxation Controls ---
    final_tau_s = 100.0   # Total slow time to evolve the atomic field
    
    # Parameters for the fast optical field relaxation at each atomic step
    relaxation_params = {'d_tau_fast': 0.00025, 'max_steps': 1000, 'tolerance': 1e-5}

    # --- F) Adaptive Step-Size Control (NEW) ---
    # Set use_adaptive_step to True to use the RKF45 method.
    use_adaptive_step = True
    
    # -- If using fixed step size (use_adaptive_step = False) --
    d_tau_s_fixed = 0.0005      # Fixed step size for the slow atomic evolution (original RK2 method).
    
    # -- If using adaptive step size (use_adaptive_step = True) --
    rkf_tolerance = 1e-4        # Max allowed error per step. Smaller is more accurate but slower.
    d_tau_s_initial = 0.001     # Initial guess for the step size.
    d_tau_s_max = 0.01          # Maximum allowed step size.
    d_tau_s_min = 1e-7          # Minimum allowed step size.
    
    # --- G) Grid and Resolution ---
    N_w = 40.0             # Number of characteristic beam waists (w_s_um) that fit in the grid width
    plot_window_um = 400.0 # Width of the visible plot window in microns. Set to 0 to disable.
    
    if pump_profile.lower() in ['lg', 'tophat']:
        Nx, Ny = 512, 512
    else:
        Nx, Ny = 384, 384
    
    Lx_um = N_w * w_s_um
    Ly_um = N_w * w_s_um
    
    print(f"Selected '{pump_profile}' profile. Grid: {Lx_um:.1f}x{Ly_um:.1f} um ({N_w} waists), {Nx}x{Ny} resolution.")

    # --- H) Initial Field Amplitudes & Radii ---
    F_amp_dimless = 3.0      # Dimensionless pump amplitude
    Psi_amp_dimless = 1.0    # Dimensionless initial BEC amplitude
    
    w_L_um = 200.0           # Physical waist of the pump beam (for LG or TopHat) in microns.
    w_BEC_um = 200.0        # Physical radius of the BEC (for TF) in microns.

    # --- I) Profile-Specific Parameters ---
    pump_lg_l, pump_lg_p, pump_lg_l_prof = 2, 0, 2
    pump_th_steepness, pump_th_l_prof = 1.0, 0

    # --- J) Miscellaneous & Data Saving Controls ---
    use_absorbing_boundaries = True
    boundary_extent, boundary_grad = 0.6, 1.0
    overwrite, track_to_max = 1, 1
    
    data_fac = final_tau_s / 200
    plot_fac = final_tau_s / 400

    # --- K) Output Directory Name ---
    output_dir_name = f"B1_Adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    #==========================================================================
    # --- END OF USER CONTROL PANEL ---
    #==========================================================================

    # --- Parameter Initialization ---
    nl_params = direct_nl_params.copy() if not use_physical_params else calculate_dimensionless_parameters(physical_params)
    
    k_L_pump = 2 * np.pi / wavelength_m
    w_s = w_s_um * 1e-6
    nl_params['beta_dd'] = 2 / (3 * k_L_pump**2 * w_s**2)
    print(f"Using direct dimensionless parameters. Calculated beta_dd = {nl_params['beta_dd']:.4e}")
    
    plot_dir, csv_dir = setup_output_directories(output_dir_name, overwrite)
    p_str = f"th{nl_params['theta']:.2f}_s{nl_params['s']}_bC{nl_params['beta_col']:.2f}_F{F_amp_dimless:.1f}".replace('.', 'p')
    
    # Log all simulation parameters
    all_params_log = {
        'Run Info': '---', 'Method': 'Relaxation', 'Output Directory': output_dir_name,
        'Parameter Style': 'Direct' if not use_physical_params else 'Calculated from Physics',
        'Scaling Waist w_s (um)': w_s_um, 'Wavelength (m)': wavelength_m,
        'Parameter Suffix': p_str, 'Dimensionless Params': '---', **nl_params,
        'Relaxation Params': '---', **relaxation_params,
        'Grid & Time': '---', 'Nx, Ny': f"{Nx}, {Ny}", 'Lx, Ly (um)': f"{Lx_um, Ly_um}",
        'Plot Window (um)': plot_window_um, 'final_tau_s': final_tau_s,
        'Time Stepping': '---', 'use_adaptive_step': use_adaptive_step,
    }
    if use_adaptive_step:
        all_params_log.update({
            'rkf_tolerance': rkf_tolerance, 'd_tau_s_initial': d_tau_s_initial,
            'd_tau_s_max': d_tau_s_max, 'd_tau_s_min': d_tau_s_min
        })
    else:
        all_params_log['d_tau_s_fixed'] = d_tau_s_fixed
    
    all_params_log.update({
        'Saving Control': '---', 'plot_fac': plot_fac, 'data_fac': data_fac,
        'overwrite': overwrite, 'track_to_max': track_to_max,
        'Boundary Control': '---', 'use_absorbing_boundaries': use_absorbing_boundaries,
        'boundary_extent': boundary_extent, 'boundary_grad': boundary_grad,
        'Initial Fields': '---', 'F_amp_dimless': F_amp_dimless, 'Psi_amp_dimless': Psi_amp_dimless,
        'w_L_um': w_L_um, 'w_BEC_um': w_BEC_um,
        'pump_lg_l': pump_lg_l, 'pump_lg_p': pump_lg_p, 'pump_lg_l_prof': pump_lg_l_prof,
        'pump_th_steepness': pump_th_steepness, 'pump_th_l_prof': pump_th_l_prof,
    })
    save_simulation_info(plot_dir, csv_dir, all_params_log, finished=False)

    # --- Grid & Wavenumber Setup ---
    x_um = np.linspace(-Lx_um/2, Lx_um/2, Nx, endpoint=False, dtype=np.float32)
    y_um = np.linspace(-Ly_um/2, Ly_um/2, Ny, endpoint=False, dtype=np.float32)
    X_um, Y_um = np.meshgrid(x_um, y_um)
    dx_um, dy_um = Lx_um/Nx, Ly_um/Ny
    
    kx_um_inv = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx_um)
    ky_um_inv = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy_um)
    Kx_um_inv, Ky_um_inv = np.meshgrid(kx_um_inv, ky_um_inv)
    K_sq_dimless = (Kx_um_inv**2 + Ky_um_inv**2) * (w_s_um**2)
    grid_params_for_plotting = {'x': x_um, 'y': y_um, 'kx': kx_um_inv, 'ky': ky_um_inv}

    # --- Initial Fields & Boundaries ---
    boundaries_mask = create_absorbing_boundaries(X_um, Y_um, x_um, Lx_um, w_s_um, boundary_extent, boundary_grad) if use_absorbing_boundaries else 1.0

    if bec_profile.lower() == 'tf': Psi0 = generate_thomas_fermi_bec(X_um, Y_um, Psi_amp_dimless, w_BEC_um, 0.01)
    elif bec_profile.lower() == 'homogeneous': Psi0 = generate_homogeneous_field(X_um, Y_um, Psi_amp_dimless, 0.01)
    else: raise ValueError(f"Unknown bec_profile: '{bec_profile}'.")

    if pump_profile.lower() == 'lg': F_P = generate_lg_pump(X_um, Y_um, F_amp_dimless, w_L_um, pump_lg_l, pump_lg_l_prof, pump_lg_p, 0.001)
    elif pump_profile.lower() == 'tophat': F_P = generate_top_hat_pump(X_um, Y_um, F_amp_dimless, w_L_um, pump_th_steepness, pump_th_l_prof, 0.001)
    elif pump_profile.lower() == 'homogeneous': F_P = generate_homogeneous_field(X_um, Y_um, F_amp_dimless, 0.001)
    else: raise ValueError(f"Unknown pump_profile: '{pump_profile}'.")

    atomic_influence = 1j * nl_params['beta_F'] * (nl_params['s'] * np.abs(Psi0)**2 - nl_params['beta_dd'] * np.abs(Psi0)**4)
    A0 = F_P / ((1.0 - 1j * nl_params['theta']) + atomic_influence + 1e-9) + generate_noise(Nx, Ny, F_amp_dimless, 0.01)

    # --- Time Stepping Preparation ---
    Psi_fft, A_steady = np.fft.fft2(Psi0), A0.copy()
    lin_op_a_relax = np.exp((-(1.0 + 1j * nl_params['theta']) - 1j * nl_params['alpha_F'] * K_sq_dimless) * (relaxation_params['d_tau_fast'] / 2.0))
    d_tau_s = d_tau_s_initial if use_adaptive_step else d_tau_s_fixed

    # --- Simulation Loop ---
    current_tau_s, iteration = 0.0, 0
    next_plot_tau = plot_fac if plot_fac > 0 else float('inf')
    next_data_tau = data_fac if data_fac > 0 else float('inf')
    tracked_maxs = np.zeros(3)

    print("\nStarting simulation loop (Relaxation Method)...")
    if use_adaptive_step: print("Using adaptive RKF45 time stepping.")
    else: print("Using fixed RK2 time stepping.")
    
    base_fname = f"state_t{current_tau_s:.2f}_{p_str}".replace('.', 'p')
    if plot_fac > 0: tracked_maxs = save_all_plots(Psi0, A_steady, F_P, plot_dir, base_fname, grid_params_for_plotting, track_to_max, tracked_maxs, plot_window_um)
    if data_fac > 0: save_field_data_csv(Psi0, csv_dir, base_fname, 'Psi'); save_field_data_csv(A_steady, csv_dir, base_fname, 'A')
    
    # Main evolution loop
    while current_tau_s < final_tau_s:
        # --- Start of Split-Step for Atomic Field ---
        if use_adaptive_step:
            step_accepted = False
            while not step_accepted:
                # 1. First linear half-step
                lin_op_psi = np.exp((-1j * K_sq_dimless) * (d_tau_s / 2.0))
                Psi_fft_temp = Psi_fft * lin_op_psi
                Psi_for_nl = np.fft.ifft2(Psi_fft_temp)

                # 2. Nonlinear full-step using RKF45
                Psi_after_nl, A_steady_final, error = rkf45_step_psi(
                    Psi_for_nl, A_steady, d_tau_s, F_P, nl_params, lin_op_a_relax, relaxation_params
                )
                
                # 3. Check error and adapt step size
                d_tau_s_suggestion = 0.9 * d_tau_s * (rkf_tolerance / (error + 1e-12))**0.2
                d_tau_s_suggestion = np.clip(d_tau_s_suggestion, d_tau_s_min, d_tau_s_max)
                
                if error <= rkf_tolerance:
                    step_accepted = True
                    # Apply second linear half-step
                    Psi_fft_after_nl = np.fft.fft2(Psi_after_nl * boundaries_mask)
                    Psi_fft = Psi_fft_after_nl * lin_op_psi
                    A_steady = A_steady_final
                    
                    if iteration % 50 == 0: print(f"Progress: Iter {iteration}, τ_s = {current_tau_s:.3f}, dt = {d_tau_s:.2e}, err = {error:.2e}")
                    
                    current_tau_s += d_tau_s
                    d_tau_s = d_tau_s_suggestion # Use suggestion for next step
                else:
                    # Reject step, retry with smaller step size
                    d_tau_s = d_tau_s_suggestion
        else: # Original fixed-step RK2 method
            lin_op_psi = np.exp((-1j * K_sq_dimless) * (d_tau_s / 2.0))
            Psi_fft *= lin_op_psi
            Psi_curr = np.fft.ifft2(Psi_fft)
            
            if iteration % 50 == 0: print(f"Progress: Iteration {iteration}, τ_s = {current_tau_s:.2f} / {final_tau_s}")
            
            A_steady_k1 = relax_optical_field_to_steady_state(Psi_curr, A_steady, F_P, nl_params, lin_op_a_relax, relaxation_params)
            k1_psi = calculate_psi_derivative(Psi_curr, A_steady_k1, nl_params)
            Psi_mid = Psi_curr + 0.5 * d_tau_s * k1_psi
            
            A_steady_k2 = relax_optical_field_to_steady_state(Psi_mid, A_steady_k1, F_P, nl_params, lin_op_a_relax, relaxation_params)
            k2_psi = calculate_psi_derivative(Psi_mid, A_steady_k2, nl_params)

            Psi_next = Psi_curr + d_tau_s * k2_psi
            A_steady = A_steady_k2
            
            Psi_fft = np.fft.fft2(Psi_next * boundaries_mask)
            Psi_fft *= lin_op_psi
            current_tau_s += d_tau_s
        # --- End of Split-Step for Atomic Field ---
        
        iteration += 1

        if np.any(np.isnan(Psi_fft)):
            print(f"ERROR: Instability at τ_s = {current_tau_s:.3f}. Aborting."); break

        # --- Data Saving ---
        if (plot_fac > 0 and current_tau_s >= next_plot_tau) or (data_fac > 0 and current_tau_s >= next_data_tau):
            Psi_snap, A_snap = np.fft.ifft2(Psi_fft), A_steady
            base_fname = f"state_t{current_tau_s:.2f}_{p_str}".replace('.', 'p')

            if plot_fac > 0 and current_tau_s >= next_plot_tau:
                tracked_maxs = save_all_plots(Psi_snap, A_snap, F_P, plot_dir, base_fname, grid_params_for_plotting, track_to_max, tracked_maxs, plot_window_um)
                next_plot_tau += plot_fac
            
            if data_fac > 0 and current_tau_s >= next_data_tau:
                save_field_data_csv(Psi_snap, csv_dir, base_fname, 'Psi')
                save_field_data_csv(A_snap, csv_dir, base_fname, 'A')
                next_data_tau += data_fac
            
    # --- Final state processing ---
    print(f"\nSimulation complete. Final slow time τ_s = {current_tau_s:.2f}")
    Psi_final, A_final = np.fft.ifft2(Psi_fft), A_steady
    base_fname = f"final_state_t{current_tau_s:.2f}_{p_str}".replace('.', 'p')
    
    save_all_plots(Psi_final, A_final, F_P, plot_dir, base_fname, grid_params_for_plotting, track_to_max, tracked_maxs, plot_window_um)
    save_field_data_csv(Psi_final, csv_dir, base_fname, 'Psi'); save_field_data_csv(A_final, csv_dir, base_fname, 'A')
    save_simulation_info(plot_dir, csv_dir, all_params_log, finished=True)
    
    print(f"\nFinal state saved. All outputs are in the '{output_dir_name}' directory.")

if __name__ == "__main__":
    main()
