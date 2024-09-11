## HEAT MAP
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Data for CLEAN
# data_clean = {
#     'Class': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'MACRO'],
#     'Accuracy': [0.9867, 0.8667, 0.8800, 0.9111],
#     'F1': [0.9796, 0.8000, 0.8235, 0.8679]
# }

# # Data for NOISY
# data_noisy = {
#     'Class': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'MACRO'],
#     'Accuracy': [0.9800, 0.8600, 0.8667, 0.9022],
#     'F1': [0.9697, 0.7921, 0.8000, 0.8540]
# }


# # Create DataFrames
# df_clean = pd.DataFrame(data_clean).set_index('Class')
# df_noisy = pd.DataFrame(data_noisy).set_index('Class')

# # Combine DataFrames
# df_combined = pd.concat([df_clean, df_noisy], axis=1, keys=['CLEAN', 'NOISY'])

# # Plot heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df_combined, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5, linecolor='black')

# # Add title and labels
# plt.title('Classification Accuracy and F1 Iris')
# plt.xlabel('Dataset')
# plt.ylabel('Class')

# # Display the plot
# plt.show()


##BAR GRAPH, CLEAN VS. NOISY
# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# datasets = ['Breast Cancer', 'House Votes', 'Iris', 'Soy Beans', 'Glass']
# f1_scores_clean = [0.9660896522, 0.8984728275, 0.8679468242, 0.9805240793, 0.8011]
# f1_scores_noisy = [0.9569, 0.8921, 0.854, 0.9434, 0.6497]

# # Number of datasets
# num_datasets = len(datasets)

# # Bar width
# bar_width = 0.35

# # Positions of bars on the x-axis
# index = np.arange(num_datasets)

# # Create bar plots
# plt.figure(figsize=(12, 7))
# bars1 = plt.bar(index - bar_width/2, f1_scores_clean, bar_width, label='F1-Score (Clean)', color='skyblue')
# bars2 = plt.bar(index + bar_width/2, f1_scores_noisy, bar_width, label='F1-Score (Noisy)', color='salmon')

# # Adding titles and labels
# plt.xlabel('Datasets')
# plt.ylabel('F1-Score')
# plt.title('F1-Score Comparison: Clean vs Noisy')
# plt.xticks(index, datasets, rotation=45, ha='right')
# plt.legend()

# # Show grid
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Display the plot
# plt.tight_layout()
# plt.show()




# #BAR GRAPH NUM OF FEATURES
# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# datasets = ['Breast Cancer', 'House Votes', 'Iris', 'Soy Beans', 'Glass']
# num_rows = [699, 435, 150, 47, 214]
# num_features = [10, 16, 4, 35, 10]
# f1_scores_clean = [0.9660896522, 0.8984728275, 0.8679468242, 0.9805240793, 0.8011]
# f1_scores_noisy = [0.9569, 0.8921, 0.854, 0.9434, 0.6497]

# # Number of datasets
# num_datasets = len(datasets)

# # Bar width
# bar_width = 0.35

# # Positions of bars on the x-axis
# index = np.arange(num_datasets)

# # Create bar plots
# plt.figure(figsize=(14, 7))

# bars1 = plt.bar(index - bar_width/2, f1_scores_clean, bar_width, label='F1-Score (Clean)', color='skyblue')
# bars2 = plt.bar(index + bar_width/2, f1_scores_noisy, bar_width, label='F1-Score (Noisy)', color='salmon')

# # Adding titles and labels
# plt.xlabel('Dataset')
# plt.ylabel('F1-Score')
# plt.title('F1-Score Comparison vs Number of Features')
# plt.xticks(index, datasets, rotation=45, ha='right')
# plt.legend()

# # Show grid
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Add feature count annotations
# for i, feature_count in enumerate(num_features):
#     plt.text(index[i] - bar_width/2, max(f1_scores_clean[i], f1_scores_noisy[i]) + 0.02,
#              f'Features: {feature_count}', ha='center', va='bottom')
    
# for i, feature_count in enumerate(num_rows):
#     plt.text(index[i] - bar_width/2, max(f1_scores_clean[i], f1_scores_noisy[i]) + 0.03,
#              f'Rows: {feature_count}', ha='center', va='top')

# # Display the plot
# plt.tight_layout()
# plt.show()



## Rows vs F1
# # Import necessary libraries
# import pandas as pd
# import matplotlib.pyplot as plt

# # Data: replace with your actual data
# data = {
#     'Features': ['Breast Cancer', 'House Votes', 'Iris', 'Soy Beans', 'Glass'],
#     'Rows': [699, 435, 150, 47, 214],
#     'F1-Score(Clean)': [0.9660896522, 0.8984728275, 0.8679468242, 0.9805240793, 0.8011],
#     'F1-Score(Noisy)': [0.9569, 0.8921, 0.854, 0.9434, 0.6497]
# }

# # Create a DataFrame from the data
# df = pd.DataFrame(data)

# # Extract data for plotting
# rows = df['Rows']
# features = df['Features']
# f1_clean = df['F1-Score(Clean)']
# f1_noisy = df['F1-Score(Noisy)']

# # Create the plot
# plt.figure(figsize=(12, 6))

# # Plot F1-Score(Clean)
# plt.plot(rows, f1_clean, marker='o', linestyle='-', color='b', label='F1-Score (Clean)')

# # Plot F1-Score(Noisy)
# plt.plot(rows, f1_noisy, marker='o', linestyle='--', color='r', label='F1-Score (Noisy)')

# # Set the x-ticks to be the row values and customize the x-tick labels
# plt.xticks(ticks=rows, labels=[f"{feature}\n({row})" for feature, row in zip(features, rows)], rotation=45, ha='right')

# # Add titles and labels
# plt.title('F1 Scores vs. Number of Rows')
# plt.xlabel('Dataset by number of Rows')
# plt.ylabel('F1 Score')
# plt.grid(True)
# plt.legend()  # Show legend to differentiate between the two lines

# # Display the plot
# plt.tight_layout()
# plt.show()



