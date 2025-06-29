# PH550-Supplementary Material
PH550 Project Repost Supplementary Material
# KerrDTS and BEC DTS are the scrpts used to simulate the models described by the PRL report.


# ----------
# Adaptive Step size utilisation
RFK45 adaptive time step can be enabled by changed the toggle in BECDTS.py, and setting to "False" reverts the code back
to fixed step size with relaxation method as described by the Model handbook by G.W.Henderson.
# -----------

# -----------
# Exel Tables
There are 3 excel tables, 2 for paramters sweep and 1 with the parameters and their corresponding patterns.
# -----------



# Quick Guide
# -----------
# Numeriocal Stability and Otical Saturation in BEC Cavity Model
you may strugle with 
numerical stability with BEC Cavity simulations, be  ware of the Sigma value (optical saturatuion), and try out different once until 
numerical stability is reached where the different interactions are balanced and stable patterns will arise/anneal with a high enough precision 
simulation run. Highly recommend running all simulations with 512x512 grid for suffcient stability and resolution.
# -----------


# -----------
# Solver fine tuning
For Relaxation parameters, I'ts recommended to set tolerance to smaller than 1e-5 but using 1e-4 to obtain a rough picture of the pattern on set before commting to run the long 
simulaiton with higher numerical precision.
# -----------
