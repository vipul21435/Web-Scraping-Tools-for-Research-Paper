import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(0, 1)  # Start from 0

# Define the equations for the lines starting from the origin
y1 = 0.5 * x  # Higher slope line starting from the origin
y2 = 1 * x  # Lower slope line starting from the origin

# Plot the lines
plt.plot(x, y1, label='Unphosphorylated Enzyme')
plt.plot(x, y2, label='Phosphorylated Enzyme')


# Adding labels and title
plt.xlabel('1 / Substrate conc. [1/μM]')
plt.ylabel('1 / Reaction rate [1/(μmol/min)]')
plt.title('Enzyme Reaction Rates vs. Substrate Concentration')
plt.legend()

# Display the plot
plt.show()
