import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

petal_width = []
with open("petal_width.csv", 'r') as file:
    for line in file:
        petal_width.append(float(line))

    file.close()

sepal_width = []
with open("sepal_width.csv", 'r') as file:
    for line in file:
        sepal_width.append(float(line))
    file.close()

# Create the dataset as a DataFrame
data = {
    'sepal length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9],
    'sepal width': sepal_width,
    'petal length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 6.1, 6.1, 5.4, 5.1, 5.1, 5.9, 6.2, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 6.1, 5.9, 6.1, 5.6, 6.3],
    'petal width': petal_width,
    'class': ['Iris-setosa']*50 + ['Iris-versicolor']*50 + ['Iris-virginica']*50
}

df = pd.DataFrame(data)

# Initialize the figure
plt.figure(figsize=(12, 8))

# List of feature names (excluding the class column)
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Loop over each feature to create a single boxplot with overlaid colored scatter points
for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)

    # Create a single boxplot for the feature (ignoring the class)
    sns.boxplot(y=feature, data=df, color='lightgray', showfliers=False)

    # Overlay scatter points colored by class
    sns.stripplot(y=feature, x='class', data=df, hue='class', jitter=True, dodge=False, palette='Set2', size=4)

    # Calculate the quartiles for the feature
    Q1 = df[feature].quantile(0.25)
    Q2 = df[feature].quantile(0.50)  # Median
    Q3 = df[feature].quantile(0.75)

    # Add horizontal lines for the quartiles
    plt.axhline(Q1, color='black', linestyle='--', linewidth=1, label='Q1')
    plt.axhline(Q2, color='black', linestyle='--', linewidth=1, label='Median (Q2)')
    plt.axhline(Q3, color='black', linestyle='--', linewidth=1, label='Q3')

    # Add labels for each quartile positioned below the respective line
    offset = 0.05  # Adjust this value for fine-tuning label positions

    plt.text(0.02, Q1 - offset, 'Q1', color='black', verticalalignment='top', fontsize=10, fontweight='bold')
    plt.text(0.02, Q2 - offset, 'Q2', color='black', verticalalignment='top', fontsize=10, fontweight='bold')
    plt.text(0.02, Q3 - offset, 'Q3', color='black', verticalalignment='top', fontsize=10, fontweight='bold')

    # Add Q4 label at the top, just below the highest value
    max_value = df[feature].max()
    plt.text(0.02, max_value - offset, 'Q4', color='black', verticalalignment='top', fontsize=10, fontweight='bold')

    plt.title(f'Boxplot with Scatter and Quartiles for {feature}')
    plt.xlabel("")  # Remove x-labels for clarity
    plt.legend([], [], frameon=False)  # Remove duplicate legends

# Show the plot
plt.tight_layout()
plt.show()
