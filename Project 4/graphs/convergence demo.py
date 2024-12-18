import matplotlib.pyplot as plt
from root_data import *
from genetic_algorthim import *
from differential_evolution import *
from PSO import *

import pandas as pd

import pandas as pd


def read_csv_into_arrays_limited(csv_file, max_rows=3000):

    # Load CSV file into a DataFrame, skipping the first row and limiting rows
    data = pd.read_csv(csv_file, skiprows=1, nrows=max_rows)

    # Convert each column to a list and group into a larger list
    grouped_arrays = [data[col].tolist() for col in data.columns]

    return grouped_arrays


def plot_convergence(convergence, num_hidden_layers, chart_title, x_label, y_label, file_name, figure_num):

    for i, fitness_values in enumerate(convergence):
        plt.plot(fitness_values, label=f'{num_hidden_layers[i]} Hidden Layers')
    plt.title(rf"$\bf{{Figure\ {figure_num}}}$: {chart_title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()


def main():
    hyperparameters = {
        'num_hidden_nodes': 2,
        'learning_rate': 0.01,
        'population_size': 7,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'scaling_factor': 0.5,
        'binomial_crossover_probability': 0.5,
        'inertia': 0.5,
        'cognitive_update_rate': 0.5,
        'social_update_rate': 0.5,
    }

    num_hidden_layers = [0, 1, 2]
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data"]

    csv_files = [
        "breast-cancer-wisconsin.csv",
        "glass.csv",
        "soybean-small.csv",
        "forestfires.csv",
        "machine.csv",
        "abalone.csv"
    ]

    datasets = [(classification, True), (regression, False)]
    algorithms = {
        "Genetic Algorithm": GeneticAlgorithm,
        "Differential Evolution": DifferentialEvolution,
        "Particle Swarm Optimization": PSO
    }

    vertical_axis_title = "Average Population Fitness"
    horizontal_axis_title = "Generations"
    file_num = 19

    # for data_group, is_class in datasets:
    #     for file in data_group:
    #         print(file)
    #         for algorithm_name, algorithm_class in algorithms.items():
    #             print(algorithm_name)
    #             convergence = []
    #             for size in num_hidden_layers:
    #                 print(size)
    #                 data = RootData(path=file, is_class=is_class)
    #                 algorithm = algorithm_class(data=data, number_hidden_layers=size, hold_out_fold=10,
    #                                             hyperparameters=hyperparameters)
    #                 fitness_values = algorithm.train()
    #                 convergence.append(fitness_values)
    #
    #             # Generate and save the chart
    #             file_name = f"figure_{file_num}_{data.name.replace(' ', '_')}_{algorithm_name.replace(' ', '_')}.png"
    #             plot_convergence(
    #                 convergence,
    #                 num_hidden_layers,
    #                 f"{data.name} {algorithm_name} Convergence",
    #                 horizontal_axis_title,
    #                 vertical_axis_title,
    #                 file_name,
    #                 file_num
    #             )
    #             file_num += 1
    #
    # Titles and axis labels
    vertical_axis_title = "Loss Value"
    horizontal_axis_title = "Epochs"
    for filename in csv_files:
        array = read_csv_into_arrays_limited(filename)
        chart_title = f"{filename.split('.')[0]} Backpropagation Convergence"
        file_name = f"figure_{file_num}_{filename.split('.')[0]}_Backpropagation.png"

        plot_convergence(
            array,
            num_hidden_layers,
            chart_title,
            horizontal_axis_title,
            vertical_axis_title,
            file_name,
            file_num
        )

        file_num += 1

if __name__ == "__main__":
    main()
