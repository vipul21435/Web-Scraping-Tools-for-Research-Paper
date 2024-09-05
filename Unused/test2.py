import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data
substrate_concentration = np.array([0.08, 0.12, 0.54, 1.23, 1.82, 2.72, 4.94, 10.00])
reaction_rate = np.array([0.15, 0.21, 0.70, 1.10, 1.30, 1.50, 1.70, 1.80])

# Extracted parameters
km_fit = 0.9
vmax_fit = 1.8

# Plot the experimental data
plt.scatter(substrate_concentration, reaction_rate, label='Experimental Data')

# Connect the data points up to Km
plt.plot(substrate_concentration, reaction_rate, linestyle='--', color='blue')

# Plot the line from Km to the y-axis at Vmax/2
plt.plot([km_fit, km_fit], [0, vmax_fit/2], color='black', linestyle='--', label = 'Km')

# Plot Vmax/2 line
plt.plot([0, km_fit], [vmax_fit/2, vmax_fit/2], color='r', linestyle='--', label='Vmax/2')

# Plot Vmax line connecting to the y-axis at Vmax/2
plt.plot([km_fit, 0], [vmax_fit/2, vmax_fit/2], color='r', linestyle='--')

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
