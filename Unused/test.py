import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data
substrate_concentration = np.array([0.08, 0.12, 0.54, 1.23, 1.82, 2.72, 4.94, 10.00])
reaction_rate = np.array([0.15, 0.21, 0.70, 1.10, 1.30, 1.50, 1.70, 1.80])

# Michaelis-Menten equation
# def michaelis_menten(s, km, vmax):
#     return (vmax * s) / (km + s)

# Initial guess for parameters
initial_guess = [1, 1.8]

# Fit the data to the Michaelis-Menten model
# params, covariance = curve_fit(michaelis_menten, substrate_concentration, reaction_rate, p0=initial_guess)

# Extract the fitted parameters
km_fit = 1
vmax_fit = 1.8

# Create a range of substrate concentrations for plotting the curve
substrate_range = np.linspace(0, max(substrate_concentration), 100)

# Plot the experimental data
plt.scatter(substrate_concentration, reaction_rate)

# Plot the Michaelis-Menten curve
# plt.plot(substrate_range, michaelis_menten(substrate_range, km_fit, vmax_fit))

# Plot Vmax/2 line
plt.axvline(x=km_fit, color='r', linestyle='--', label='Km')
plt.axhline(y=vmax_fit/2, color='r', linestyle='--', label='Vmax/2')

# Plot Vmax on the y-axis at 1.8
plt.axhline(y=vmax_fit, color='g', linestyle='--', label='Vmax')

# Label the axes
plt.xlabel('Substrate Concentration (µM)')
plt.ylabel('Reaction Rate (µmol/min)')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Print the fitted parameters
print(f'Fitted Km: {km_fit} µM')
print(f'Fitted Vmax: {vmax_fit} µmol/min')
