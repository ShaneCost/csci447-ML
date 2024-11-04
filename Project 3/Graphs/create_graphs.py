import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define confusion matrices
confusion_matrices = {
    '0_hidden_layers': np.array([[5, 4, 11, 7, 16, 17],
                                  [10, 8, 11, 10, 16, 15],
                                  [1, 5, 2, 1, 3, 4],
                                  [1, 1, 3, 1, 2, 2],
                                  [1, 4, 1, 0, 2, 1],
                                  [6, 6, 7, 5, 2, 2]]),
    
    '1_hidden_layer': np.array([[49, 11, 0, 0, 0, 0],
                                 [7, 56, 0, 5, 2, 0],
                                 [0, 15, 1, 0, 0, 0],
                                 [0, 4, 0, 4, 0, 2],
                                 [0, 1, 0, 0, 3, 5],
                                 [0, 2, 0, 1, 0, 25]]),
    
    '2_hidden_layers': np.array([[36, 24, 0, 0, 0, 0],
                                  [30, 38, 0, 1, 0, 1],
                                  [5, 11, 0, 0, 0, 0],
                                  [2, 2, 1, 1, 0, 4],
                                  [0, 1, 0, 1, 1, 6],
                                  [0, 1, 0, 6, 0, 21]])
}

# Create heatmaps and save them as PDF files
for title, matrix in confusion_matrices.items():
    plt.figure(figsize=(6, 6))  # Set figure size
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', cbar=False,
                 xticklabels=['D1', 'D2', 'D3', 'D4'], yticklabels=['D1', 'D2', 'D3', 'D4'])
    plt.title(f'Confusion Matrix: {title.replace("_", " ")}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the heatmap as a PDF

    plt.savefig(f'glass,{title}.pdf', format='pdf')
    plt.close()  # Close the figure to free up memory

print("Heatmaps saved as PDF files.")
