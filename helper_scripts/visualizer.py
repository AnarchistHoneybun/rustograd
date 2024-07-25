import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('moon_dataset.csv')
boundary = pd.read_csv('decision_boundary_data2.csv')

# Create the plot
plt.figure(figsize=(10, 8))

# Unique values for the transformed decision boundary
x_unique = boundary['x'].unique()
y_unique = boundary['y'].unique()

# Reshape z values for contour plot
z_values = boundary['z'].values.reshape(len(x_unique), len(y_unique))

# Plot the decision boundary
plt.contourf(x_unique, y_unique, z_values, levels=0, cmap='RdBu_r', alpha=0.9)
plt.contour(x_unique, y_unique, z_values, levels=0, colors=['yellow'], linewidths=2)

# Plot the data points
plt.scatter(data[data['y'] == -1]['x1'], data[data['y'] == -1]['x2'], c='blue', label='Class -1')
plt.scatter(data[data['y'] == 1]['x1'], data[data['y'] == 1]['x2'], c='red', label='Class 1')

plt.legend()
plt.title('MLP Decision Boundary on Moon Dataset')
plt.xlabel('X')
plt.ylabel('Y')

# Retrieve current axes limits
x_lim = plt.xlim()
y_lim = plt.ylim()

# Plot dotted axes using the current limits
plt.plot(x_lim, [0, 0], color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.plot([0, 0], y_lim, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.savefig('mlp_decision_boundary.png')
plt.show()
