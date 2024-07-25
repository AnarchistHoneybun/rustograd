import numpy as np
from sklearn.datasets import make_moons
import csv

# Generate the moon-shaped dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Convert labels from 0/1 to -1/1
y = y * 2 - 1

# Save the dataset to a CSV file
with open('../moon_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x1', 'x2', 'y'])  # Header
    for (x1, x2), label in zip(X, y):
        writer.writerow([x1, x2, label])

print("Dataset generated and saved to 'moon_dataset.csv'")