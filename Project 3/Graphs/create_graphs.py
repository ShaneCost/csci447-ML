import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a custom color map that goes from white to red
cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

# Confusion Matrices for each hidden layer configuration

# Hidden layer 0
cm0 = np.array([
    [2, 1, 6, 1],
    [3, 1, 1, 5],
    [3, 4, 0, 3],
    [3, 7, 2, 1]
])

# Hidden layer 1
cm1 = np.array([
    [10, 0, 0, 0],
    [0, 10, 0, 0],
    [0, 0, 10, 0],
    [0, 0, 1, 12]
])

# Hidden layer 2
cm2 = np.array([
    [7, 2, 1, 0],
    [1, 8, 0, 1],
    [1, 0, 8, 1],
    [0, 0, 3, 10]
])

# Create and save heatmap for the hidden layer 0 confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm0, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=["D1", "D2", "D3", "D4"], yticklabels=["D1", "D2", "D3", "D4"], cbar_kws={'label': 'Counts'})
plt.title('Soybean-small: hidden layers 0', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('Soybean-small_0.pdf')  # Save as a PDF
plt.close()  # Close the figure to avoid overlapping with the next one

# Create and save heatmap for the hidden layer 1 confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm1, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=["D1", "D2", "D3", "D4"], yticklabels=["D1", "D2", "D3", "D4"], cbar_kws={'label': 'Counts'})
plt.title('Soybean-small: hidden layers 1', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('Soybean-small_1.pdf')  # Save as a PDF
plt.close()

# Create and save heatmap for the hidden layer 2 confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm2, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=["D1", "D2", "D3", "D4"], yticklabels=["D1", "D2", "D3", "D4"], cbar_kws={'label': 'Counts'})
plt.title("Soybean-small: hidden layers 0", fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('Soybean-small_2.pdf')  # Save as a PDF
plt.close()
