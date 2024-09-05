import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Data
substrate_conc = [12.5, 8.3, 1.9, 0.8, 0.5, 0.4, 0.2, 0.1]
reaction_rate = [6.7, 4.8, 1.4, 0.9, 0.8, 0.7, 0.6, 0.6]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(substrate_conc, reaction_rate)
line_reg = [slope * x + intercept for x in substrate_conc]

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot of data points
plt.scatter(substrate_conc, reaction_rate, color='blue', label='Data Points')

# Linear regression line
plt.plot(substrate_conc, line_reg, color='red', linestyle='dotted', label='Linear Regression Line')

# Line connecting the dots
plt.plot(substrate_conc, reaction_rate, color='green', linestyle='solid', marker='o', label='Line Connecting Dots')

plt.title('Enzyme Reaction Rate vs Substrate Concentration')
plt.xlabel('1/Substrate Concentration [1/μM]')
plt.ylabel('1/Reaction Rate [1/(μmol/min)]')
plt.grid(True)
plt.legend()
plt.show()
