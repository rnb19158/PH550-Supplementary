#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEC Pattern Analysis Tool
Created on Fri May 30 18:11:51 2025

@author: Zhanming Mei

This script analyses the output data from the BEC-Cavity simulation (BECDTS.py).
It takes the CSV files representing the complex atomic field (Psi) and performs
two key analyses as per the user's request:

1.  Lattice Constant (Λ) Calculation:
    - Computes the 2D Fast Fourier Transform (FFT) of the atomic intensity |Psi|**2.
    - Identifies the primary peaks in the Fourier spectrum, which correspond to the
      fundamental spatial frequencies of the pattern.
    - Calculates the lattice constant Λ = 2*pi/|k|, where |k| is the wavevector
      magnitude of the pattern.
    - Estimates uncertainty from the standard deviation of the identified peak distances.

2.  Rotational Order Parameter (Psi_4 and Psi_6) Calculation:
    - Finds local maxima (i.e., the bright spots) in the real-space intensity pattern.
    - Calculates the four-fold (Psi_4) and six-fold (Psi_6) rotational order
      parameters to quantitatively measure the pattern's symmetry.
    - Classifies the pattern as hexagonal, square, or disordered based on
      the calculated order parameter values.

Usage:
    python analyze_bec_pattern.py /path/to/simulation/output_directory [--plot]
"""

import numpy as np
import os
import glob
import argparse
import re
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from matplotlib.patches import Circle

# --- Core Analysis Functions ---

def calculate_lattice_constant(intensity, dx_um):
    """
    Calculates the lattice constant from the intensity pattern's Fourier transform.
    The lattice constant is the characteristic spacing of the pattern in real space.

    Args:
        intensity (np.ndarray): The 2D real-space intensity pattern, |Psi|^2.
        dx_um (float): The grid spacing in the x-direction, in micrometres. This is
                       vital for converting from pixel space to physical units.

    Returns:
        tuple: A tuple containing:
            - float: The calculated average lattice constant (Λ) in micrometres.
            - float: The uncertainty in the lattice constant.
            - list: Pixel coordinates of the identified peaks in Fourier space.
    """
    if np.all(intensity == 0):
        return 0, 0, []

    Ny, Nx = intensity.shape

    # 1. Fourier Transform
    # We compute the 2D FFT to move from real space (x, y) to Fourier space (kx, ky).
    # The peaks in Fourier space represent the dominant spatial frequencies of the pattern.
    # fftshift moves the zero-frequency component (DC term) to the centre of the array.
    fft_field = np.fft.fftshift(np.fft.fft2(intensity))
    power_spectrum = np.abs(fft_field)**2

    # Define k-space coordinates. This creates a grid of wavevector values (kx, ky)
    # that corresponds to the grid of the power spectrum.
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_um)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx_um)) * 2 * np.pi

    # 2. Peak Identification
    # The central DC component corresponds to the mean intensity, not the pattern itself,
    # so we exclude it from the peak search by masking a small radius around the centre.
    centre_x, centre_y = Nx // 2, Ny // 2
    dc_mask_radius = 5  # pixels
    Y, X = np.ogrid[:Ny, :Nx]
    dist_from_centre = np.sqrt((X - centre_x)**2 + (Y - centre_y)**2)
    power_spectrum[dist_from_centre <= dc_mask_radius] = 0

    # Find local maxima in the 2D power spectrum. A peak is a point that is brighter
    # than all its neighbours. We use a threshold of 50% of the max non-DC value
    # to filter out noise and find only the most significant peaks.
    peak_threshold = 0.5 * power_spectrum.max()
    local_max = maximum_filter(power_spectrum, size=15) # size is the filter window
    peak_mask = (power_spectrum == local_max) & (power_spectrum > peak_threshold)
    peak_coords = np.argwhere(peak_mask) # Get the (row, column) of each peak

    if len(peak_coords) < 2:
        print("Warning: Could not find sufficient peaks in Fourier space to determine lattice constant.")
        return 0, 0, []

    # 3. Wavevector Measurement
    # Convert peak pixel coordinates (py, px) to physical k-space coordinates and
    # then calculate the magnitude of the wavevector |k| for each peak.
    k_vectors = [(kx[px], ky[py]) for py, px in peak_coords]
    k_magnitudes = [np.sqrt(kx_val**2 + ky_val**2) for kx_val, ky_val in k_vectors]
    
    avg_k = np.mean(k_magnitudes)
    if avg_k == 0:
        return 0, 0, peak_coords

    # 4. Calculation
    # The lattice constant, Λ, is inversely related to the wavevector magnitude.
    lattice_constant = 2 * np.pi / avg_k

    # The uncertainty is derived from the standard deviation of the measured peak distances.
    # This gives a measure of how consistently spaced the peaks are (i.e., how perfect the lattice is).
    k_std_dev = np.std(k_magnitudes)
    # Propagate the error using the formula: delta_L = |-2pi/k^2| * delta_k
    uncertainty = (2 * np.pi / avg_k**2) * k_std_dev

    return lattice_constant, uncertainty, peak_coords

def calculate_order_parameters(intensity):
    """
    Calculates the four-fold and six-fold rotational order parameters. These
    parameters measure how closely the pattern resembles a perfect square or hexagon.

    Args:
        intensity (np.ndarray): The 2D real-space intensity pattern, |Psi|^2.

    Returns:
        tuple: A tuple containing:
            - float: The six-fold order parameter (Ψ₆). Value is near 1 for hexagonal patterns.
            - float: The four-fold order parameter (Ψ₄). Value is near 1 for square patterns.
            - list: Coordinates of the identified intensity peaks in real space.
    """
    if np.all(intensity == 0):
        return 0, 0, []

    Ny, Nx = intensity.shape
    
    # 1. Peak Identification
    # Identify local intensity maxima in real space (the bright spots of the pattern).
    # We only consider peaks above 50% of the global maximum intensity.
    peak_threshold = 0.5 * intensity.max()
    local_max = maximum_filter(intensity, size=10) # size is the filter window
    peak_mask = (intensity == local_max) & (intensity > peak_threshold)
    peak_coords = np.argwhere(peak_mask)
    
    N = len(peak_coords) # Total number of identified peaks
    if N < 2:
        return 0, 0, []

    # 2. Coordinate Conversion
    # Convert Cartesian pixel coordinates to polar coordinates (r, φ) relative to
    # the centre of the grid. We only need the angle (phi) for this calculation.
    centre_x, centre_y = Nx / 2.0, Ny / 2.0
    x_coords = peak_coords[:, 1] - centre_x
    y_coords = peak_coords[:, 0] - centre_y
    
    # np.arctan2(y, x) computes the angle φ for each peak.
    angles = np.arctan2(y_coords, x_coords)

    # 3. Complex Summation
    # For a perfectly N-fold symmetric pattern, the vectors exp(i*N*φ) for each peak
    # will point in the same direction. When summed, they add up constructively. For a
    # disordered pattern, they point in random directions and the sum is close to zero.
    
    # Six-fold order parameter (for hexagonal symmetry)
    psi6_sum = np.sum(np.exp(6j * angles))
    psi6 = np.abs(psi6_sum) / N

    # Four-fold order parameter (for square symmetry)
    psi4_sum = np.sum(np.exp(4j * angles))
    psi4 = np.abs(psi4_sum) / N

    return psi6, psi4, peak_coords

# --- Helper and I/O Functions ---

def load_atomic_field(base_path):
    """
    Loads the real and imaginary parts of a field from CSV files and combines
    them into a single complex numpy array.

    Args:
        base_path (str): The base path of the CSV files, without the _real/_imag suffix.

    Returns:
        np.ndarray: A complex 2D numpy array representing the field, or None if files not found.
    """
    real_path = f"{base_path}_real.csv"
    imag_path = f"{base_path}_imag.csv"

    if not os.path.exists(real_path) or not os.path.exists(imag_path):
        print(f"Error: Could not find data files for base path: {base_path}")
        return None

    real_part = np.loadtxt(real_path, delimiter=',')
    imag_part = np.loadtxt(imag_path, delimiter=',')
    
    return real_part + 1j * imag_part

def parse_simulation_info(info_file_path):
    """
    Parses the simulation_info.txt file to extract key grid parameters.
    This is essential for converting pixel-based measurements into physical units.

    Args:
        info_file_path (str): Path to the simulation_info.txt file.

    Returns:
        dict: A dictionary with parameters like 'Lx_um' and 'Nx', or None on failure.
    """
    params = {}
    try:
        with open(info_file_path, 'r') as f:
            for line in f:
                # Use regex to find key-value pairs like 'Lx, Ly (um)' or 'Nx, Ny'
                match = re.search(r"^\s*([\w\s,\(\)]+?)\s*=\s*(.+)$", line)
                if match:
                    key, value = match.groups()
                    key = key.strip()
                    value = value.strip()
                    if key == 'Lx, Ly (um)':
                        # Extracts the first value for Lx
                        lx_val = float(value.strip('()').split(',')[0])
                        params['Lx_um'] = lx_val
                    elif key == 'Nx, Ny':
                        # Extracts the first value for Nx
                        nx_val = int(value.split(',')[0])
                        params['Nx'] = nx_val
    except FileNotFoundError:
        print(f"Error: simulation_info.txt not found at {info_file_path}")
        return None
    except Exception as e:
        print(f"Error parsing simulation info file: {e}")
        return None
        
    if 'Lx_um' not in params or 'Nx' not in params:
        print("Error: Could not find 'Lx, Ly (um)' or 'Nx, Ny' in info file.")
        return None
        
    return params

def create_summary_plot(intensity, real_peaks, k_space_spectrum, k_peaks, analysis_results, output_path):
    """
    Generates and saves a summary plot of the analysis, showing both real and
    Fourier space representations of the pattern.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('default')

    # --- Plot 1: Real Space Intensity ---
    # This shows the actual pattern of atoms, |Psi|^2.
    ax1 = axes[0]
    im1 = ax1.imshow(intensity, cmap='viridis', origin='lower')
    ax1.set_title("Real-Space Intensity $|\\Psi|^2$")
    # Overlay red circles on the identified intensity peaks.
    ax1.scatter(real_peaks[:, 1], real_peaks[:, 0], facecolors='none', edgecolors='r', s=80, label='Identified Peaks')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # --- Plot 2: Fourier Space Power Spectrum ---
    # This shows the pattern's representation in frequency space.
    ax2 = axes[1]
    # Use a logarithmic scale for better visualisation of peaks against the background.
    log_spectrum = np.log10(k_space_spectrum + 1e-12)
    im2 = ax2.imshow(log_spectrum, cmap='magma', origin='lower')
    ax2.set_title("Fourier-Space Power Spectrum (log scale)")
    # Draw circles on identified k-space peaks to highlight them.
    for r, c in k_peaks:
        ax2.add_patch(Circle((c, r), radius=5, color='cyan', fill=False, linewidth=2))
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # --- Add a text box with the final calculated results ---
    text_content = (
        f"Analysis Results:\n"
        f"-----------------\n"
        f"Lattice Constant (Λ): {analysis_results['Lambda']:.2f} ± {analysis_results['Lambda_err']:.2f} µm\n"
        f"Order Parameter (Ψ₄): {analysis_results['Psi4']:.3f}\n"
        f"Order Parameter (Ψ₆): {analysis_results['Psi6']:.3f}\n"
        f"Classification: {analysis_results['Classification']}"
    )
    fig.text(0.5, 0.02, text_content, ha='center', va='bottom', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for text box
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved analysis plot to: {output_path}")

# --- Main Execution Block ---

def main():
    """Main function to orchestrate the analysis."""
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Analyse BEC-Cavity simulation data.")
    parser.add_argument("directory", type=str, help="Path to the simulation output directory.")
    parser.add_argument("--plot", action="store_true", help="Generate and save summary plots for each analysed file.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at {args.directory}")
        return

    # Find and parse the simulation info file to get essential grid parameters.
    info_file = os.path.join(args.directory, "simulation_info.txt")
    sim_params = parse_simulation_info(info_file)
    if sim_params is None:
        return
    # Calculate the grid spacing, which is crucial for physical unit conversion.
    dx_um = sim_params['Lx_um'] / sim_params['Nx']
    print(f"Grid parameters loaded: Lx = {sim_params['Lx_um']} µm, Nx = {sim_params['Nx']}, dx = {dx_um:.3f} µm")

    # Find all atomic field data files in the specified 'csv_data' subdirectory.
    csv_dir = os.path.join(args.directory, 'csv_data')
    search_pattern = os.path.join(csv_dir, "*_Psi_real.csv")
    psi_files = glob.glob(search_pattern)

    if not psi_files:
        print(f"No Psi data files found in {csv_dir}")
        return

    print(f"\nFound {len(psi_files)} Psi data file(s) to analyse...")

    # Loop through each found data file and perform the analysis.
    for real_file_path in sorted(psi_files):
        base_name = os.path.basename(real_file_path).replace('_real.csv', '')
        base_path = os.path.join(csv_dir, base_name)
        print(f"\n--- Analysing: {base_name} ---")

        # Load the complex atomic field Psi from the CSV files.
        psi_field = load_atomic_field(base_path)
        if psi_field is None:
            continue
        
        # The intensity is the squared magnitude of the complex field.
        intensity = np.abs(psi_field)**2

        # --- Perform Analyses ---
        
        # 1. Calculate the Lattice Constant.
        k_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(intensity)))**2
        lattice_const, uncertainty, k_peaks = calculate_lattice_constant(intensity, dx_um)
        
        # 2. Calculate the Order Parameters.
        psi6, psi4, real_peaks = calculate_order_parameters(intensity)

        # 3. Classify the pattern based on the order parameter thresholds.
        if psi6 > 0.8:
            classification = "Strong Hexagonal Order"
        elif psi4 > 0.8:
            classification = "Strong Square Order"
        else:
            classification = "Disordered or Other"
            
        # --- Print Results to the Console ---
        print(f"  Lattice Constant (Λ): {lattice_const:.3f} ± {uncertainty:.3f} µm")
        print(f"  Six-Fold Order (Ψ₆):  {psi6:.4f}")
        print(f"  Four-Fold Order (Ψ₄): {psi4:.4f}")
        print(f"  Pattern Classification: {classification}")

        # --- Plotting (if requested) ---
        if args.plot:
            analysis_results = {
                'Lambda': lattice_const, 'Lambda_err': uncertainty,
                'Psi4': psi4, 'Psi6': psi6, 'Classification': classification
            }
            plot_output_path = os.path.join(args.directory, f"analysis_{base_name}.png")
            create_summary_plot(intensity, real_peaks, k_spectrum, k_peaks, analysis_results, plot_output_path)


if __name__ == "__main__":
    main()
