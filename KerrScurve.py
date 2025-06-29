#!/usr/bin/env python3
"""
Created on Mon june 2 11:37:58 2025

@author: zhanmingmei

Kerr Cavity S-curve Diagram Plotter
"""



import numpy as np
import matplotlib.pyplot as plt

# Set backend
import matplotlib
matplotlib.use('Agg')

# Create figure
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# ===============================================
# S-Curve for different cavity detunings
# ===============================================

def calculate_s_curve_simple(theta, Fp_max=4, n_points=200):
    """Calculate S-curve for given detuning - simplified version"""
    Fp_values = np.linspace(0, Fp_max, n_points)
    F_lower = np.zeros(n_points)
    F_upper = np.zeros(n_points)
    F_middle = np.zeros(n_points)
    
    for i, Fp in enumerate(Fp_values):
        if Fp == 0:
            F_lower[i] = 0
            F_upper[i] = np.nan
            F_middle[i] = np.nan
            continue
        
        # For the steady state equation: |F|² = |Fp|²/[1 + (θ - |F|²)²]
        # Define function to find roots
        F_test = np.linspace(0, 5, 1000)
        equation = F_test**2 * (1 + (theta - F_test**2)**2) - Fp**2
        
        # Find sign changes (roots)
        sign_changes = np.where(np.diff(np.sign(equation)))[0]
        
        if len(sign_changes) == 0:
            # No exact solution, use approximation
            F_lower[i] = Fp / np.sqrt(1 + theta**2)
            F_upper[i] = np.nan
            F_middle[i] = np.nan
        elif len(sign_changes) == 1:
            # One solution
            sol = F_test[sign_changes[0]]
            if theta <= np.sqrt(3) or Fp < 1.5:
                F_lower[i] = sol
                F_upper[i] = np.nan
                F_middle[i] = np.nan
            else:
                F_lower[i] = np.nan
                F_upper[i] = sol
                F_middle[i] = np.nan
        elif len(sign_changes) >= 3:
            # Three solutions (bistable)
            F_lower[i] = F_test[sign_changes[0]]
            F_middle[i] = F_test[sign_changes[1]]
            F_upper[i] = F_test[sign_changes[2]]
        else:
            # Two solutions - handle carefully
            if F_test[sign_changes[0]] < theta/2:
                F_lower[i] = F_test[sign_changes[0]]
                F_upper[i] = F_test[sign_changes[1]]
                F_middle[i] = np.nan
            else:
                F_lower[i] = np.nan
                F_upper[i] = F_test[sign_changes[1]]
                F_middle[i] = np.nan
    
    return Fp_values, F_lower, F_middle, F_upper

# Plot S-curves for different detunings
theta_values = [0.5, 1.5, np.sqrt(3), 2.5, 3.5]
colors = ['blue', 'green', 'orange', 'red', 'purple']

for theta, color in zip(theta_values, colors):
    Fp, F_lower, F_middle, F_upper = calculate_s_curve_simple(theta)
    label = f'θ = {theta:.2f}'
    
    # Plot branches
    # Lower stable branch
    mask_lower = ~np.isnan(F_lower)
    if np.any(mask_lower):
        ax.plot(Fp[mask_lower], F_lower[mask_lower], '-', 
               color=color, linewidth=2.5, label=label)
    
    # Upper stable branch
    mask_upper = ~np.isnan(F_upper)
    if np.any(mask_upper):
        ax.plot(Fp[mask_upper], F_upper[mask_upper], '-', 
               color=color, linewidth=2.5)
    
    # Middle unstable branch
    mask_middle = ~np.isnan(F_middle)
    if np.any(mask_middle) and theta > np.sqrt(3):
        ax.plot(Fp[mask_middle], F_middle[mask_middle], '--', 
               color=color, linewidth=2, alpha=0.5)

# Add critical detuning line
ax.axvline(x=np.sqrt(3), color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(np.sqrt(3)+0.05, 3.5, 'θ = √3\n(Critical)', fontsize=13, ha='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

# Add annotations
# --- CHANGE: Moved "Monostable" label further left ---
ax.text(0.8, 0.7, 'Monostable', fontsize=16, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
# --- CHANGE: Increased "Bistable" font size ---
ax.text(3.0, 2.5, 'Bistable', fontsize=16, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))

# Hysteresis arrows
ax.annotate('', xy=(2.5, 3.2), xytext=(2.5, 0.6),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))
ax.annotate('', xy=(1.2, 0.4), xytext=(1.2, 2.8),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=3))
# --- CHANGE: Increased ON/OFF font size ---
ax.text(2.6, 1.8, 'ON', fontsize=15, color='darkgreen', weight='bold')
ax.text(0.95, 1.5, 'OFF', fontsize=15, color='darkorange', weight='bold')

# Formatting
# --- CHANGE: Increased axis label and title font size ---
ax.set_xlabel('Input Field |Fp|', fontsize=25, weight='bold')
ax.set_ylabel('Intracavity Field |F|', fontsize=25, weight='bold')
ax.set_title('Optical Bistability: S-Curves for Different Kerr Cavity Detunings', 
             fontsize=25, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':')
# --- CHANGE: Increased legend font size ---
ax.legend(loc='upper left', fontsize=14)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

plt.tight_layout()
plt.savefig('s_curve_fixed.png', dpi=200, bbox_inches='tight')
plt.savefig('s_curve_fixed.pdf', bbox_inches='tight')
print("Saved S-curve plot as s_curve_fixed.png and .pdf")
plt.close()

# ===============================================
# Simplified single S-curve demonstration
# ===============================================

fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)

# Show clear S-curve for θ = 3.0
theta_demo = 3.0
Fp_demo = np.linspace(0, 3.5, 300)
F_demo = np.zeros_like(Fp_demo)

# Calculate steady state
for i, Fp in enumerate(Fp_demo):
    if Fp < 0.01:
        F_demo[i] = 0
        continue
    
    # Numerical solution
    F_test = np.linspace(0, 4, 500)
    residual = np.abs(F_test**2 * (1 + (theta_demo - F_test**2)**2) - Fp**2)
    min_idx = np.argmin(residual)
    
    # For bistable region, choose appropriate branch
    if Fp < 1.3:  # Lower branch
        F_demo[i] = F_test[min_idx]
    elif Fp > 2.3:  # Upper branch
        F_test_upper = F_test[F_test > 2]
        if len(F_test_upper) > 0:
            residual_upper = np.abs(F_test_upper**2 * (1 + (theta_demo - F_test_upper**2)**2) - Fp**2)
            F_demo[i] = F_test_upper[np.argmin(residual_upper)]
        else:
            F_demo[i] = F_test[min_idx]
    else:  # Transition region
        F_demo[i] = np.nan

# Plot
valid = ~np.isnan(F_demo)
ax2.plot(Fp_demo[valid], F_demo[valid], 'b-', linewidth=3, label='Stable')

# Add unstable branch approximation
Fp_unstable = np.linspace(1.3, 2.3, 20)
F_unstable = theta_demo/2 * np.ones_like(Fp_unstable)  # Approximate
ax2.plot(Fp_unstable, F_unstable, 'r--', linewidth=2.5, label='Unstable', alpha=0.7)

# Hysteresis
ax2.annotate('', xy=(2.3, 3.0), xytext=(2.3, 0.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax2.annotate('', xy=(1.3, 0.4), xytext=(1.3, 2.5),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))

ax2.text(2.4, 1.5, 'Switch ON', fontsize=12, color='green', weight='bold', rotation=90)
ax2.text(1.15, 1.4, 'Switch OFF', fontsize=12, color='orange', weight='bold', rotation=90)

ax2.set_xlabel('Pump Field |Fp|', fontsize=14, weight='bold')
ax2.set_ylabel('Intracavity Field |F|', fontsize=14, weight='bold')
ax2.set_title(f'Classic S-Curve Bistability (θ = {theta_demo})', fontsize=16, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=12)
ax2.set_xlim(0, 3.5)
ax2.set_ylim(0, 3.5)

plt.tight_layout()
plt.savefig('s_curve_simple.png', dpi=200, bbox_inches='tight')
plt.savefig('s_curve_simple.pdf', bbox_inches='tight')
print("Saved simple S-curve as s_curve_simple.png and .pdf")
plt.close()

print("\n✓ Fixed dimension mismatch issue!")
print("Generated S-curve plots showing optical bistability")
