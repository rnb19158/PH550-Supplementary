#!/usr/bin/env python3
"""
Created on Sat June 7 15:18:22 2025

@author: Zhanming Mei

Description:
This script generates and saves a morphology phase diagram,
visualising different phases based on two parameters: cavity detuning (theta)
and a collisional parameter (beta_col).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# --- Aesthetic Configuration ---

# Configure professional fonts and colours for the plot.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Georgia', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# Define a colour palette for the different phases and plot elements.
colours = {
    'Homogeneous': '#87CEFA',  # LightSkyBlue
    'Hexagonal':   '#F4A460',  # SandyBrown
    'Square':      '#3CB371',  # MediumSeaGreen
    'Complex':     '#DDA0DD',  # Plum
    'Localised':   '#B0C4DE',  # LightSteelBlue
    'Borders':     '#2F4F4F',  # DarkSlateGray
    'Uncertainty': '#808080'   # Gray for uncertainty bands
}
phase_colours = [colours['Homogeneous'], colours['Hexagonal'], colours['Square'], colours['Complex'], colours['Localised']]

# --- Main Plotting Logic ---

# Initialise the figure and axes for the plot.
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Define the parameter space by creating a meshgrid. Extended theta range on both sides.
theta = np.linspace(-3, 7, 400)
beta_col = np.linspace(-2.5, 2.5, 400)
THETA, BETA_COL = np.meshgrid(theta, beta_col)

# --- Phase Region Definitions ---

# Initialise the phase array, defaulting to the 'Localised' phase (index 4).
phase = np.full_like(THETA, 4)

# Define each phase region using logical masks.
phase[THETA < 0] = 0  # Homogeneous

mask_hex = (THETA >= 0) & (THETA <= 2.5) & (np.abs(BETA_COL) <= 1.5)
phase[mask_hex] = 1   # Hexagonal

mask_square = (THETA >= 0) & (THETA <= 3.5) & (np.abs(BETA_COL) > 1.5)
phase[mask_square] = 2 # Square

mask_complex = (THETA > 2.5) & (THETA <= 4.5) & (np.abs(BETA_COL) <= 1.5)
phase[mask_complex] = 3 # Complex

# Render the phase regions on the plot with updated extent.
cmap = plt.cm.colors.ListedColormap(phase_colours)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
ax.imshow(phase, extent=[-3, 7, -2.5, 2.5], aspect='auto', origin='lower',
          cmap=cmap, norm=norm, interpolation='antialiased')

# Draw contour lines to delineate the phase boundaries.
ax.contour(THETA, BETA_COL, phase, levels=[0.5, 1.5, 2.5, 3.5],
           colors=colours['Borders'], linewidths=2.2)

# --- Uncertainty Band Visualisation ---

# Define uncertainty values for each parameter.
theta_uncertainty = 0.1
beta_uncertainty = 0.05

# Add vertical uncertainty bands for theta boundaries.
ax.add_patch(Rectangle((-theta_uncertainty, -2.5), 2 * theta_uncertainty, 5,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10)) # Boundary at theta = 0
ax.add_patch(Rectangle((2.5 - theta_uncertainty, -1.5), 2 * theta_uncertainty, 3,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10)) # Boundary at theta = 2.5
ax.add_patch(Rectangle((4.5 - theta_uncertainty, -1.5), 2 * theta_uncertainty, 3,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10)) # Boundary at theta = 4.5

# Add horizontal uncertainty bands for beta_col boundaries.
ax.add_patch(Rectangle((0, 1.5 - beta_uncertainty), 3.5, 2 * beta_uncertainty,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10)) # Boundary at beta_col = +1.5
ax.add_patch(Rectangle((0, -1.5 - beta_uncertainty), 3.5, 2 * beta_uncertainty,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10)) # Boundary at beta_col = -1.5

# Add uncertainty at the corner where Square and Localised phases meet.
ax.add_patch(Rectangle((3.5 - theta_uncertainty, 1.5 - beta_uncertainty), 2 * theta_uncertainty, 2 * beta_uncertainty,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10))
ax.add_patch(Rectangle((3.5 - theta_uncertainty, -1.5 - beta_uncertainty), 2 * theta_uncertainty, 2 * beta_uncertainty,
                       facecolor=colours['Uncertainty'], alpha=0.3, zorder=10))

# --- Text Labels and Annotations ---

# Define positions for text labels to identify each phase region.
labels = {
    'Homogeneous':   {'pos': (-1.5, 0)},
    'Hexagonal':     {'pos': (1.25, 0)},
    'Square_top':    {'pos': (1.75, 2.0), 'label': 'Square'},
    'Square_bot':    {'pos': (1.75, -2.0),'label': 'Square'},
    'Complex':       {'pos': (3.5, 0)},
    'Localised_mid': {'pos': (5.75, 0), 'label': 'Localised'}, # Re-centred the labels
    'Localised_top': {'pos': (5.75, 2.0),'label': 'Localised'},
    'Localised_bot': {'pos': (5.75, -2.0),'label': 'Localised'}
}

# Place the labels on the plot with larger font size.
for key, val in labels.items():
    label_text = val.get('label', key.split('_')[0])
    ax.text(val['pos'][0], val['pos'][1], label_text,
            fontsize=22, weight='bold', color=colours['Borders'],
            ha='center', va='center')

# Set plot labels, title, grid, and axis limits.
ax.set_xlabel(r'Cavity Detuning $\theta$', fontsize=35, weight='bold', labelpad=15)
ax.set_ylabel(r'Collisional Parameter $\beta_{col}$', fontsize=35, weight='bold', labelpad=15)
ax.set_title('Morphology Phase Diagram', fontsize=35, weight='bold', pad=25)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.set_xlim(-3, 7) # Updated x-axis limit
ax.set_ylim(-2.5, 2.5)
ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=6)

# Construct the legend with larger fonts.
legend_elements = [
    Patch(facecolor=colours['Homogeneous'], edgecolor=colours['Borders'], label='Homogeneous'),
    Patch(facecolor=colours['Hexagonal'],   edgecolor=colours['Borders'], label='Hexagonal'),
    Patch(facecolor=colours['Square'],      edgecolor=colours['Borders'], label='Square'),
    Patch(facecolor=colours['Complex'],     edgecolor=colours['Borders'], label='Complex'),
    Patch(facecolor=colours['Localised'],   edgecolor=colours['Borders'], label='Localised'),
    Patch(facecolor=colours['Uncertainty'], alpha=0.3, edgecolor='none',
          label=f'Uncertainty:\n'
                f'$\\pm${theta_uncertainty} in $\\theta$\n'
                f'$\\pm${beta_uncertainty} in $\\beta_{{col}}$')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=20, frameon=True, fancybox=True, shadow=False,
          title='Phase', title_fontsize=22)

# Add a text box detailing the uncertainty values directly on the plot with larger font.
ax.text(0.02, 0.02,
        f'Phase boundaries: $\\Delta\\theta = \\pm${theta_uncertainty}, '
        f'$\\Delta\\beta_{{col}} = \\pm${beta_uncertainty}',
        transform=ax.transAxes, fontsize=16,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Adjust layout to prevent clipping and save the figure in multiple formats.
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('morphology_phase_diagram_with_uncertainty.png', dpi=300, bbox_inches='tight')
plt.savefig('morphology_phase_diagram_with_uncertainty.pdf', bbox_inches='tight')

print("Phase diagrams with uncertainty bands saved as 'morphology_phase_diagram_with_uncertainty.png' and .pdf")
