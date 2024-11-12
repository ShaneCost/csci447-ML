import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a custom color map that goes from white to red
cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

# Confusion Matrices from the provided data

# First confusion matrix
cm1 = np.array([
    [11, 9, 8, 7, 18, 7],
    [17, 8, 13, 14, 11, 7],
    [5, 5, 2, 2, 1, 1],
    [3, 2, 0, 3, 1, 3],
    [5, 0, 1, 1, 0, 2],
    [8, 3, 3, 4, 5, 3]
])

# Second confusion matrix
cm2 = np.array([
    [47, 7, 0, 2, 7, 0],
    [20, 21, 13, 6, 4, 6],
    [1, 7, 3, 2, 2, 1],
    [0, 2, 1, 1, 2, 4],
    [0, 0, 1, 1, 1, 5],
    [0, 1, 0, 0, 0, 25]
])

# Third confusion matrix
cm3 = np.array([
    [11, 34, 6, 6, 6, 0],
    [7, 41, 7, 7, 7, 1],
    [1, 10, 2, 1, 2, 0],
    [1, 4, 1, 1, 0, 3],
    [0, 5, 1, 0, 1, 1],
    [1, 12, 1, 0, 1, 11]
])

# Create and save heatmap for the first confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm1, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=[1, 2, 3, 5, 6, 7], yticklabels=[1, 2, 3, 5, 6, 7], cbar_kws={'label': 'Counts'})
plt.title('glass_hidden_layers_0', fontsize=14)  # Update title
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('glass_hidden_layers_0.png')  # Save the first heatmap as a PNG with an updated file name
plt.close()  # Close the figure to avoid overlapping with the next one

# Create and save heatmap for the second confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm2, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=[1, 2, 3, 5, 6, 7], yticklabels=[1, 2, 3, 5, 6, 7], cbar_kws={'label': 'Counts'})
plt.title('glass_hidden_layers_1', fontsize=14)  # Update title
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('glass_hidden_layers_1.png')  # Save the second heatmap as a PNG with an updated file name
plt.close()

# Create and save heatmap for the third confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm3, annot=True, fmt="d", cmap=cmap, cbar=True, 
            xticklabels=[1, 2, 3, 5, 6, 7], yticklabels=[1, 2, 3, 5, 6, 7], cbar_kws={'label': 'Counts'})
plt.title('glass_hidden_layers_2', fontsize=14)  # Update title
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('glass_hidden_layers_2.png')  # Save the third heatmap as a PNG with an updated file name
plt.close()
