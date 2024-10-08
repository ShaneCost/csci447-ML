import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define confusion matrices
matrices = {
    'Breast_Cancer_KMeans': np.array([[398, 16], [9, 207]]),
    'Glass_KMeans': np.array([[59, 2, 0, 0, 0, 0],
                                 [8, 47, 14, 1, 0, 0],
                                 [0, 7, 8, 0, 1, 0],
                                 [0, 3, 0, 3, 2, 2],
                                 [0, 0, 3, 2, 2, 2],
                                 [0, 0, 2, 1, 1, 23]]),
    'Soybean_Small_KMeans': np.array([[5, 0, 2, 3],
                                        [0, 7, 0, 3],
                                        [1, 0, 4, 5],
                                        [0, 0, 1, 12]]),
    'Breast_Cancer_KNN': np.array([[402, 12], [10, 206]]),
    'Glass_KNN': np.array([[64, 1, 0, 0, 0, 0],
                            [1, 65, 0, 4, 0, 0],
                            [0, 3, 13, 0, 0, 0],
                            [0, 1, 0, 7, 1, 1],
                            [0, 0, 0, 1, 6, 1],
                            [0, 0, 1, 2, 2, 19]]),
    'Soybean_Small_KNN': np.array([[10, 0, 0, 0],
                                    [0, 10, 0, 0],
                                    [0, 0, 10, 0],
                                    [0, 0, 0, 13]]),
    'Breast_Cancer_Edited_KNN': np.array([[403, 11], [10, 206]]),
    'Glass_Edited_KNN': np.array([[60, 0, 0, 0, 0, 0],
                                    [4, 55, 9, 1, 1, 0],
                                    [0, 2, 13, 0, 0, 0],
                                    [0, 7, 0, 2, 0, 3],
                                    [0, 2, 0, 0, 5, 2],
                                    [0, 0, 0, 1, 4, 22]]),
    'Soybean_Small_Edited_KNN': np.array([[10, 0, 0, 0],
                                           [0, 8, 0, 2],
                                           [1, 0, 9, 0],
                                           [0, 0, 0, 13]])
}

# Class labels for each matrix
labels = {
    'Breast_Cancer': ['2', '4'],
    'Glass': ['1', '2', '3', '5', '6', '7'],
    'Soybean_Small': ['D1', 'D2', 'D3', 'D4']
}

# Create heatmaps and save each as a file
for title, matrix in matrices.items():
    plt.figure(figsize=(8, 6))
    
    # Determine the correct label set
    if 'Breast_Cancer' in title:
        label_set = labels['Breast_Cancer']
    elif 'Glass' in title:
        label_set = labels['Glass']
    elif 'Soybean_Small' in title:
        label_set = labels['Soybean_Small']
    
    # Create heatmap with white to red color map
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds", cbar=True, 
                 xticklabels=label_set, yticklabels=label_set)
    
    plt.title(title.replace('_', ' ').title())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the figure
    plt.savefig(f"{title}.pdf")
    plt.close()  # Close the figure to avoid display

print("Heatmaps saved as individual files.")
