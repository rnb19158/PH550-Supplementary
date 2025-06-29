#!/usr/bin/env python3
"""
Created on Fri May 13 17:23:48 2025

@author: Zhanming Mei

Kerr Cavity Simulator with Adaptive Step-Size (RKF45)

This script simulates the 2D Lugiato-Lefever Equation (LLE) for a Kerr
cavity using a split-step Fourier method. This version incorporates a
Runge-Kutta-Fehlberg 45 (RKF45) method for the nonlinear step, allowing
for an adaptive (dynamic) time step size. This ensures numerical convergence
within a user-defined error tolerance, the script is built following the methodology 
described in the Model handbook by G. W. Henderson.

(v4 - Adaptive Step-Size):
- Replaced the 4th-order Runge-Kutta (RK4) integrator for the nonlinear
  step with an adaptive RKF45 method.
- Introduced 'dt_initial' and 'rkf_threshold' parameters to control the
  dynamic step-size algorithm.
- The time step 'dt' is now dynamically adjusted during the simulation
  to maintain the truncation error below the specified threshold.
- All original utility functions and the output structure are retained.

(v2 - Decoupled Radius):
- Introduced w_L to independently control the physical radius of LG and TopHat pumps.
- The pump beam's radius is no longer tied to the grid size (Lx).

(v1):
- Updated I/O handling to create a single, timestamped parent directory for each run.
- All simulation parameters are logged to a 'simulation_info.txt' file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.special import assoc_laguerre

###############################################################################
# SECTION 1: UTILITY & SIMULATION FUNCTIONS 
###############################################################################

def generate_noise(Nx, Ny, amp, noise_amplitude, dtype=np.complex64):
    """
    Creates a field of random complex noise.
    """
    noise_re = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    noise_im = np.random.uniform(-1, 1, (Ny, Nx)).astype(np.float32)
    return (amp * noise_amplitude * (noise_re + 1j * noise_im)).astype(dtype)

def calculate_spectrum_log(field):
    """
    Calculates the spatial power spectrum of the field in log scale.
    """
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    return np.log10(spec + 1e-9) # Added a small epsilon to avoid log(0)

def setup_output_directories(base_directory):
    """
    Creates the base output directory and subdirectories for plots and CSV data.
    """
    plots_dir = os.path.join(base_directory, 'plots')
    csv_data_dir = os.path.join(base_directory, 'csv_data')

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_data_dir, exist_ok=True)

    print(f"Created output directory: {os.path.abspath(base_directory)}")
    return base_directory, plots_dir, csv_data_dir

def save_field_plots(A, directory, base_filename, x_phys, y_phys, kx_vals, ky_vals, k_zoom_factor=1.5):
    """
    Saves plots of the field's amplitude, phase, and log-spectrum to a single PNG file,
    styled to remove axes and labels, retaining only the colour bar.
    """
    amp = np.abs(A)
    phs = np.angle(A) + np.pi # Shift phase from [-pi, pi] to [0, 2pi]
    log_spec = calculate_spectrum_log(A)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3)

    # --- Amplitude plot ---
    im1 = axes[0].imshow(amp, origin='lower',
                         extent=[x_phys.min(), x_phys.max(), y_phys.min(), y_phys.max()],
                         cmap='viridis')
    axes[0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_ticks([np.min(amp), np.max(amp)])
    cbar1.set_ticklabels(['min', 'max'])

    # --- Phase plot ---
    im2 = axes[1].imshow(phs, origin='lower', cmap='hsv',
                         extent=[x_phys.min(), x_phys.max(), y_phys.min(), y_phys.max()],
                         vmin=0, vmax=2*np.pi)
    axes[1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_ticks([0, 2 * np.pi])
    cbar2.set_ticklabels(['0', '2π'])

    # --- Log-Spectrum plot ---
    kx_shifted = np.fft.fftshift(kx_vals)
    ky_shifted = np.fft.fftshift(ky_vals)
    im3 = axes[2].imshow(log_spec, origin='lower',
                         extent=[kx_shifted.min(), kx_shifted.max(), ky_shifted.min(), ky_shifted.max()],
                         cmap='magma')
    axes[2].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_xlim(kx_shifted.min() / k_zoom_factor, kx_shifted.max() / k_zoom_factor)
    axes[2].set_ylim(ky_shifted.min() / k_zoom_factor, ky_shifted.max() / k_zoom_factor)
    im3.set_clim(np.percentile(log_spec, 5), log_spec.max())

    # --- Save Figure ---
    full_save_path = os.path.join(directory, f"{base_filename}.png")
    plt.savefig(full_save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved plot: {full_save_path}")
    plt.close(fig)

def save_field_data_csv(A, directory, base_filename):
    """
    Saves the real and imaginary parts of the complex field A to separate CSV files.
    """
    filepath_real = os.path.join(directory, f"{base_filename}_real.csv")
    np.savetxt(filepath_real, A.real, delimiter=',')

    filepath_imag = os.path.join(directory, f"{base_filename}_imag.csv")
    np.savetxt(filepath_imag, A.imag, delimiter=',')
    print(f"Saved CSV data for: {base_filename}")


def save_simulation_info(output_dir, plots_dir, csv_data_dir, params_dict, finished=False):
    """
    Saves simulation parameters and timestamps to 'simulation_info.txt'.
    """
    info_file_path = os.path.join(output_dir, "simulation_info.txt")
    mode = 'a' if os.path.exists(info_file_path) else 'w'
    with open(info_file_path, mode) as f:
        if not finished:
            f.write(f"\nKerr Cavity Simulation (START) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*60 + "\n")
            f.write(f"Plot output:   {os.path.abspath(plots_dir)}\n")
            f.write(f"CSV data:      {os.path.abspath(csv_data_dir)}\n\n")
            f.write("Parameters:\n")
            for section, params in params_dict.items():
                f.write(f"  --- {section} ---\n")
                for key, value in params.items():
                    f.write(f"    {key:<20} = {value}\n")
            f.write("\n")
        else:
            f.write(f"\nKerr Cavity Simulation (END) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*60 + "\n")

###############################################################################
# SECTION 2: MAIN SIMULATION
###############################################################################
def main():
    # --------------------------------------------------------------
    # This is the control panel for Simulation & Physical Parameters
    # ---------------------------------------------------------------
    final_tau = 1000.0
    dt_initial = 0.01  # Initial time step, will be adapted
    rkf_threshold = 1e-5 # Max allowed error for RKF45 adaptive step
    theta = 0
    Is = 1.07
    beta_K = 1
    
    F_amp_calc = np.sqrt(Is * (1.0 + (theta - beta_K * Is)**2))
    print(f"Reference uniform pump amplitude F_amp_calc = {F_amp_calc:.4f}")

    # Grid scaling parameters
    w_s = 10.0      # Characteristic beam waist. Defines the grid size, for Transverse scaling.
    N_w = 5.0       # Number of characteristic beam waists (w_s) that fit in the grid width
    Nx, Ny = 256, 256
    
    Lx = N_w * w_s  # Grid width is now calculated
    Ly = N_w * w_s  # Assume square grid
    dx, dy = Lx / Nx, Ly / Ny
    print(f"Grid calculated: {Lx:.1f}x{Ly:.1f} dimensionless units ({N_w} waists).")


    # --- Output Control ---
    save_interval = 10.0  # Time units between saves
    k_zoom_factor = 1.0   # For zooming into the k-space plot
    output_dir_name = f"Kerr_Sim_Adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # -----------------------------
    # Pump Configuration
    # -----------------------------
    pump_type = "lg"  # Options: "homogeneous", "LG", "tophat"
    pump_F_amp = F_amp_calc
    
    w_L = 10.0 # Physical waist of the pump beam (for LG or TopHat)

    pump_lg_m = 1
    pump_lg_p = 0

    pump_th_m = 1
    pump_th_S = 1

    # ----------------------------------------
    # Create Directories & Parameter Strings
    # ----------------------------------------
    run_dir, plot_save_dir, csv_data_dir = setup_output_directories(output_dir_name)

    p_str = f"th{theta:.2f}_Is{Is:.2f}".replace('.', 'p')
    if pump_type.lower() == "lg":
        p_str += f"_pumpLG_m{pump_lg_m}_p{pump_lg_p}"
    elif pump_type.lower() == "tophat":
        p_str += f"_pumpTH_m{pump_th_m}"
    else:
        p_str += "_pumpHomogeneous"

    # -----------------------------
    # Logging
    # -----------------------------
    all_params_log = {
        'Run Info': {
            'Output Directory': output_dir_name,
            'Parameter Suffix': p_str,
        },
        'Physical Parameters': {
            "theta": theta, "Is": Is, "beta_K": f"{beta_K:.4f}",
            "Calculated F_amp": f"{F_amp_calc:.4f}"
        },
        'Numerical Parameters': {
            "final_tau": final_tau, "dt_initial": dt_initial,
            "rkf_threshold": rkf_threshold,
            "Nx, Ny": f"{Nx}, {Ny}",
            "w_s (scaling waist)": w_s, "N_w (num waists)": N_w,
            "Lx, Ly (calculated)": f"{Lx}, {Ly}", "save_interval": save_interval
        },
        'Pump Parameters': {
            "pump_type": pump_type, "pump_F_amp": f"{pump_F_amp:.4f}",
            "w_L (beam radius)": w_L,
            "LG (l, p)": f"{pump_lg_m}, {pump_lg_p}" if pump_type.lower() == "lg" else "N/A",
            "Tophat (m, S)": f"{pump_th_m}, {pump_th_S:.2f}" if pump_type.lower() == "tophat" else "N/A",
        }
    }
    save_simulation_info(run_dir, plot_save_dir, csv_data_dir, all_params_log, finished=False)

    # -----------------------------
    # Spatial Grid & Wavenumbers
    # -----------------------------
    x_phys = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False, dtype=np.float32)
    y_phys = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(x_phys, y_phys, indexing='xy')

    kx_vals = (2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)).astype(np.float32)
    ky_vals = (2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)).astype(np.float32)
    Kx, Ky = np.meshgrid(kx_vals, ky_vals, indexing='xy')
    K_sq = (Kx**2 + Ky**2).astype(np.float32)

    # -------------------------------
    # Pump Field Generation
    # -------------------------------
    pump_field = np.zeros((Ny, Nx), dtype=np.complex64)
    if pump_type.lower() == "lg":
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        laguerre_poly = assoc_laguerre(2 * R**2 / w_L**2, pump_lg_p, np.abs(pump_lg_m))
        radial_profile = (np.sqrt(2) * R / w_L)**np.abs(pump_lg_m) * laguerre_poly * np.exp(-R**2 / w_L**2)
        if np.max(np.abs(radial_profile)) > 1e-9:
            radial_profile /= np.max(np.abs(radial_profile))
        pump_field = pump_F_amp * radial_profile * np.exp(1j * pump_lg_m * Phi)
        print(f"LG pump generated. Max amplitude: {np.max(np.abs(pump_field)):.4f}")

    elif pump_type.lower() == "tophat":
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        radial_profile = 0.5 * (1 - np.tanh( pump_th_S * (R - w_L)))
        pump_field = pump_F_amp * radial_profile * np.exp(1j * pump_th_m * Phi)
        print(f"Top-hat pump generated. Max amplitude: {np.max(np.abs(pump_field)):.4f}")

    else: # Default to uniform "homogeneous"
        pump_field = np.full((Ny, Nx), pump_F_amp, dtype=np.complex64)
        print(f"Uniform (Homogeneous) pump generated. Amplitude: {pump_F_amp:.4f}")

    # ------------------------------------
    # Initial Field & Time Stepping Prep
    # ------------------------------------
    A0 = generate_noise(Nx, Ny, amp=0.01 * pump_F_amp, noise_amplitude=1.0)
    A = A0.copy()
    
    # --- Save initial state and pump profile ---
    initial_base_filename = f"initial_state_t0p00_{p_str}"
    save_field_plots(A, plot_save_dir, initial_base_filename, x_phys, y_phys, kx_vals, ky_vals, k_zoom_factor)
    save_field_data_csv(A, csv_data_dir, initial_base_filename)
    
    if np.any(pump_field):
        pump_base_filename = f"pump_profile_{p_str}"
        save_field_plots(pump_field, plot_save_dir, pump_base_filename, x_phys, y_phys, kx_vals, ky_vals, k_zoom_factor)

    # --------------------------------------------
    # Main Evolution Loop with Adaptive Step-Size
    # --------------------------------------------
    current_tau = 0.0
    iteration = 0
    next_save_time = save_interval
    dt = dt_initial
    A_fft = np.fft.fft2(A)

    print(f"\nStarting simulation... Evolving up to tau = {final_tau:.1f} with adaptive dt.")

    while current_tau < final_tau:
        # Define the nonlinear function for the RKF45 method
        def nonlinear_func(field):
            return pump_field + 1j * beta_K * np.abs(field)**2 * field

        # Store the field before the full step
        A_current_step_start = np.fft.ifft2(A_fft)

        # Adaptive step-size loop
        while True:
            # 1) First half-step linear operator in k-space
            linear_op_half_dt = np.exp(dt/2 * (-1.0 - 1j*theta - 1j*K_sq)).astype(np.complex64)
            A_fft_half = A_fft * linear_op_half_dt
            A_real_start = np.fft.ifft2(A_fft_half)

            # 2) Full nonlinear step using Runge-Kutta-Fehlberg 45 (RKF45)
            # This follows the Butcher Tableau from the Model Handbook, page 19.
            k1 = dt * nonlinear_func(A_real_start)
            k2 = dt * nonlinear_func(A_real_start + 1/4 * k1)
            k3 = dt * nonlinear_func(A_real_start + 3/32 * k1 + 9/32 * k2)
            k4 = dt * nonlinear_func(A_real_start + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k5 = dt * nonlinear_func(A_real_start + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k6 = dt * nonlinear_func(A_real_start - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)

            # 4th and 5th order solutions
            A_rk4 = A_real_start + (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
            A_rk5 = A_real_start + (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)

            # Estimate the truncation error
            error = np.max(np.abs(A_rk5 - A_rk4))
            
            # Check if the error is within the tolerance
            if error <= rkf_threshold:
                # Step is accepted. Use the more accurate 5th order solution.
                A = A_rk5
                # Calculate the optimal dt for the *next* step
                if error > 1e-12: # Avoid division by zero
                    dt_next = 0.9 * dt * (rkf_threshold / error)**(1/5)
                else:
                    dt_next = dt * 2 # If error is tiny, increase step size
                break # Exit the adaptive loop
            else:
                # Step is rejected. Reduce dt and retry the current step.
                dt = 0.9 * dt * (rkf_threshold / error)**(1/5)
                # No break, loop will run again with smaller dt

        # 3) Transform back to k-space
        A_fft = np.fft.fft2(A)

        # 4) Second half-step linear operator in k-space
        linear_op_half_dt = np.exp(dt/2 * (-1.0 - 1j*theta - 1j*K_sq)).astype(np.complex64)
        A_fft *= linear_op_half_dt
        
        # Update time and step size for next iteration
        current_tau += dt
        dt = dt_next
        iteration += 1

        # --- Progress reporting and instability check ---
        if iteration % 100 == 0:
            A_current = np.fft.ifft2(A_fft)
            maxA_abs = np.max(np.abs(A_current))
            print(f"Iter {iteration}, tau = {current_tau:.2f}/{final_tau} ({(current_tau/final_tau)*100:.1f}%), Max|A| = {maxA_abs:.4f}, dt = {dt:.2e}")
            if np.isnan(maxA_abs) or maxA_abs > 1e6:
                print(f"ERROR: Instability detected at tau = {current_tau:.3f}. Max|A| = {maxA_abs:.2e}. Aborting.")
                break

        # --- Save snapshot ---
        if current_tau >= next_save_time:
            print(f"Saving snapshot at tau = {current_tau:.2f}...")
            A_save = np.fft.ifft2(A_fft)
            
            time_str = f"{current_tau:.2f}".replace('.', 'p')
            snapshot_base_filename = f"state_t{time_str}_{p_str}"
            
            save_field_plots(A_save, plot_save_dir, snapshot_base_filename, x_phys, y_phys, kx_vals, ky_vals, k_zoom_factor)
            save_field_data_csv(A_save, csv_data_dir, snapshot_base_filename)
            
            next_save_time += save_interval
            print(f"Snapshot saved. Next save at tau >= {next_save_time:.2f}.")

    # ----------------------------------
    # End of Simulation - Final Outputs
    # ----------------------------------
    print(f"\nSimulation loop finished at tau = {current_tau:.2f}")

    A_final = np.fft.ifft2(A_fft)
    
    final_time_str = f"{current_tau:.2f}".replace('.', 'p')
    final_base_filename = f"final_state_t{final_time_str}_{p_str}"
    
    save_field_plots(A_final, plot_save_dir, final_base_filename, x_phys, y_phys, kx_vals, ky_vals, k_zoom_factor)
    save_field_data_csv(A_final, csv_data_dir, final_base_filename)
    
    save_simulation_info(run_dir, plot_save_dir, csv_data_dir, all_params_log, finished=True)

    print("\nSimulation complete.")
    print(f"All output files for this run are in: {os.path.abspath(run_dir)}")
    print(f"Max amplitude in final field = {np.max(np.abs(A_final)):.4f}")

    return A_final

if __name__ == "__main__":
    final_field = main()
