#!/usr/bin/env python3
"""
Created on Sat june 7 15:18:22 2025

@author: Zhanming Mei

Norphology Phase Diagram Plotter
"""




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Aesthetic Enhancements ---

# Set professional fonts and colors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Georgia', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

colors = {
    'Homogeneous': '#87CEFA',  # LightSkyBlue
    'Hexagonal':   '#F4A460',  # SandyBrown
    'Square':      '#3CB371',  # MediumSeaGreen
    'Complex':     '#DDA0DD',  # Plum
    'Localized':   '#B0C4DE',  # LightSteelBlue
    'Borders':     '#2F4F4F'   # DarkSlateGray
}
phase_colors = [colors['Homogeneous'], colors['Hexagonal'], colors['Square'], colors['Complex'], colors['Localized']]

# --- Main Plotting Logic ---

# Create figure
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Create parameter space
theta = np.linspace(-2, 6, 400)
beta_col = np.linspace(-2.5, 2.5, 400)
THETA, BETA_COL = np.meshgrid(theta, beta_col)

# --- CORRECTED PHASE DEFINITIONS ---
# The boundary has been moved from -0.5 to 0 to widen the Homogeneous box as requested.
# The original script used -0.5 as the boundary.

phase = np.full_like(THETA, 4)  # Default to Localized (4)

# CORRECTED: Homogeneous region now extends to theta = 0
phase[(THETA < 0)] = 0

# CORRECTED: Hexagonal region now starts from theta = 0
mask_hex = (THETA >= 0) & (THETA <= 2.5) & (np.abs(BETA_COL) <= 1.5)
phase[mask_hex] = 1

# CORRECTED: Square region now starts from theta = 0
mask_square = (THETA >= 0) & (THETA <= 3.5) & (np.abs(BETA_COL) > 1.5)
phase[mask_square] = 2

# Complex region definition remains the same relative to other phases
mask_complex = (THETA > 2.5) & (THETA <= 4.5) & (np.abs(BETA_COL) <= 1.5)
phase[mask_complex] = 3

# Plot the phase regions
cmap = plt.cm.colors.ListedColormap(phase_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
ax.imshow(phase, extent=[-2, 6, -2.5, 2.5], aspect='auto', origin='lower',
          cmap=cmap, norm=norm, interpolation='antialiased')

# Refine contour lines
ax.contour(THETA, BETA_COL, phase, levels=[0.5, 1.5, 2.5, 3.5],
           colors=colors['Borders'], linewidths=2.2)

# Position text labels within their regions
# The 'Homogeneous' label position is updated to be centered in its new, wider box.
labels = {
    'Homogeneous': {'pos': (-1.0, 0)}, # CORRECTED: Re-centered in the wider box
    'Hexagonal':   {'pos': (1.25, 0)}, # Adjusted for narrower box
    'Square_top':  {'pos': (1.75, 2.0), 'label': 'Square'}, # Adjusted
    'Square_bot':  {'pos': (1.75, -2.0),'label': 'Square'}, # Adjusted
    'Complex':     {'pos': (3.5, 0)},
    'Localised_mid':{'pos': (5.25, 0), 'label': 'Localised'},
    'Localised_top':{'pos': (5.25, 2.0),'label': 'Localised'},
    'Localised_bot':{'pos': (5.25, -2.0),'label': 'Localised'}
}

for key, val in labels.items():
    label_text = val.get('label', key.split('_')[0])
    ax.text(val['pos'][0], val['pos'][1], label_text,
            fontsize=18, weight='bold', color=colors['Borders'],
            ha='center', va='center')

# Add labels, title, grid, and ticks
ax.set_xlabel(r'Cavity Detuning $\theta$', fontsize=35, weight='bold', labelpad=15)
ax.set_ylabel(r'Collisional Parameter $\beta_{col}$', fontsize=35, weight='bold', labelpad=15)
ax.set_title('Morphology Phase Diagram', fontsize=35, weight='bold', pad=25)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.set_xlim(-2, 6)
ax.set_ylim(-2.5, 2.5)
ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=6)

# Create legend
legend_elements = [
    Patch(facecolor=colors['Homogeneous'], edgecolor=colors['Borders'], label='Homogeneous'),
    Patch(facecolor=colors['Hexagonal'],   edgecolor=colors['Borders'], label='Hexagonal'),
    Patch(facecolor=colors['Square'],      edgecolor=colors['Borders'], label='Square'),
    Patch(facecolor=colors['Complex'],     edgecolor=colors['Borders'], label='Complex'),
    Patch(facecolor=colors['Localized'],   edgecolor=colors['Borders'], label='Localized')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=20, frameon=True, fancybox=True, shadow=False,
          title='Phase', title_fontsize=16)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('morphology_phase_diagram_widened.png')
plt.savefig('morphology_phase_diagram_widened.pdf')

print("Corrected phase diagrams with widened box saved as 'morphology_phase_diagram_widened.png' and .pdf")
