import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Updated data for the glass dataset
data = {
    "Hidden Layer 0": {
        "DA": {
            "1": [2, 11, 6, 25, 7, 9],
            "2": [15, 14, 10, 16, 5, 10],
            "3": [5, 2, 2, 4, 2, 0],
            "5": [5, 3, 0, 1, 1, 3],
            "6": [3, 2, 0, 2, 0, 1],
            "7": [5, 9, 4, 0, 2, 7]
        },
        "GA": {
            "1": [8, 21, 3, 1, 1, 26],
            "2": [20, 29, 4, 1, 1, 15],
            "3": [5, 4, 1, 0, 0, 5],
            "5": [6, 3, 2, 1, 1, 0],
            "6": [4, 3, 0, 1, 0, 0],
            "7": [15, 7, 1, 2, 2, 0]
        },
        "PSO": {
            "1": [14, 19, 11, 7, 0, 9],
            "2": [19, 19, 4, 9, 0, 19],
            "3": [5, 4, 0, 1, 1, 4],
            "5": [2, 3, 0, 3, 1, 4],
            "6": [1, 2, 1, 2, 0, 2],
            "7": [7, 8, 4, 4, 0, 4]
        }
    },
    "Hidden Layer 1": {
        "DA": {
            "1": [14, 33, 1, 0, 0, 12],
            "2": [16, 43, 0, 0, 0, 11],
            "3": [7, 6, 1, 0, 0, 1],
            "5": [6, 5, 1, 0, 0, 1],
            "6": [3, 4, 1, 0, 0, 0],
            "7": [14, 11, 2, 0, 0, 0]
        },
        "GA": {
            "1": [15, 45, 0, 0, 0, 0],
            "2": [13, 57, 0, 0, 0, 0],
            "3": [3, 12, 0, 0, 0, 0],
            "5": [5, 8, 0, 0, 0, 0],
            "6": [2, 6, 0, 0, 0, 0],
            "7": [7, 20, 0, 0, 0, 0]
        },
        "PSO": {
            "1": [20, 16, 6, 5, 7, 6],
            "2": [24, 19, 12, 4, 3, 8],
            "3": [8, 4, 2, 1, 0, 0],
            "5": [7, 2, 3, 0, 1, 0],
            "6": [3, 2, 1, 1, 0, 1],
            "7": [16, 7, 4, 0, 0, 0]
        }
    },
    "Hidden Layer 2": {
        "DA": {
            "1": [13, 35, 0, 6, 0, 6],
            "2": [17, 40, 0, 6, 0, 7],
            "3": [6, 8, 0, 0, 0, 1],
            "5": [5, 6, 0, 1, 0, 1],
            "6": [5, 2, 0, 0, 0, 1],
            "7": [8, 12, 1, 2, 1, 3]
        },
        "GA": {
            "1": [0, 60, 0, 0, 0, 0],
            "2": [0, 70, 0, 0, 0, 0],
            "3": [0, 15, 0, 0, 0, 0],
            "5": [0, 13, 0, 0, 0, 0],
            "6": [0, 8, 0, 0, 0, 0],
            "7": [0, 27, 0, 0, 0, 0]
        },
        "PSO": {
            "1": [23, 24, 6, 1, 4, 2],
            "2": [25, 20, 6, 5, 5, 9],
            "3": [5, 5, 3, 1, 0, 1],
            "5": [3, 2, 3, 1, 1, 3],
            "6": [2, 3, 2, 0, 0, 1],
            "7": [4, 9, 1, 6, 2, 5]
        }
    }
}

# Function to plot confusion matrix with white to red color scheme
def plot_confusion_matrix(data, layer_name, method):
    # Get the confusion matrix for the current method (1, 2, 3, 5, 6, 7)
    matrix = np.array([
        data[layer_name][method]["1"], 
        data[layer_name][method]["2"], 
        data[layer_name][method]["3"], 
        data[layer_name][method]["5"],
        data[layer_name][method]["6"],
        data[layer_name][method]["7"]
    ])
    
    # Create a custom colormap from white to red
    cmap = plt.cm.Reds
    norm = plt.Normalize(vmin=0, vmax=np.max(matrix))
    
    # Define the labels for the axes
    labels = ["1", "2", "3", "5", "6", "7"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, cbar=False, norm=norm, 
                linewidths=0.5, xticklabels=labels, yticklabels=labels)
    plt.title(f'Glass: {layer_name} - {method}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the confusion matrix image
    plt.savefig(f"Confusion_Matrices\\Glass_{layer_name}_{method}.png")
    plt.close()

# Main function to execute the script
def main():
    # Iterate over each hidden layer and method to generate confusion matrix plots
    for layer_name in data:
        for method in data[layer_name]:
            plot_confusion_matrix(data, layer_name, method)

# Run the main function
if __name__ == "__main__":
    main()
