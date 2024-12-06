from feedforward_neural_network import *
from meta_data import *
from root_data import *
from confusion_matrix import *
from PSO import *
from genetic_algorthim import *
from differential_evolution import *
import os

tuned_hyperparameters_classification = {
    "breast-cancer-wisconsin": {
        0 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.77,
            "mutation_rate": 0.11,
            "DE_population_size": 68,
            "scaling_factor": 0.0,
            "binomial_crossover_probability": 0.0,
            "PSO_population_size": 15,
            "inertia": 0.25,
            "cognitive_update_rate": 0.93,
            "social_update_rate": 0.57
        },
        1 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.11,
            "mutation_rate": 0.0,
            "DE_population_size": 15,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 0.66,
            "PSO_population_size": 47,
            "inertia": 0.56,
            "cognitive_update_rate": 0.75,
            "social_update_rate": 1.11
        },
        2 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.0,
            "mutation_rate": 0.0,
            "DE_population_size": 5.0,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 1.0,
            "PSO_population_size": 70,
            "inertia": 0.21,
            "cognitive_update_rate": 0.87,
            "social_update_rate": 1.15
        }
    },
    "glass": {
        0 : {
            "GA_population_size": 7.0,
            "crossover_rate": 0.56,
            "mutation_rate": 0.02,
            "DE_population_size": 47,
            "scaling_factor": 0.0,
            "binomial_crossover_probability": 0.0,
            "PSO_population_size": 16,
            "inertia": 0.31,
            "cognitive_update_rate": 0.87,
            "social_update_rate": 0.64
        },
        1 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.03,
            "mutation_rate": 0.01,
            "DE_population_size": 17,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 0.32,
            "PSO_population_size": 34,
            "inertia": 0.72,
            "cognitive_update_rate": 0.47,
            "social_update_rate": 1.13
        },
        2 : {
            "GA_population_size": 4.0,
            "crossover_rate": 0.01,
            "mutation_rate": 0.11,
            "DE_population_size": 5.0,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 1.0,
            "PSO_population_size": 87,
            "inertia": 0.91,
            "cognitive_update_rate": 0.97,
            "social_update_rate": 1.09
        }
    },
    "soybean-small": {
        0 : {
            "GA_population_size": 6.0,
            "crossover_rate": 0.76,
            "mutation_rate": 0.04,
            "DE_population_size": 38,
            "scaling_factor": 0.05,
            "binomial_crossover_probability": 0.07,
            "PSO_population_size": 19,
            "inertia": 0.39,
            "cognitive_update_rate": 0.77,
            "social_update_rate": 0.55
        },
        1 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.06,
            "mutation_rate": 0.11,
            "DE_population_size": 13,
            "scaling_factor": 1.05,
            "binomial_crossover_probability": 0.87,
            "PSO_population_size": 43,
            "inertia": 0.67,
            "cognitive_update_rate": 0.38,
            "social_update_rate": 1.01
        },
        2 : {
            "GA_population_size": 9.0,
            "crossover_rate": 0.17,
            "mutation_rate": 0.21,
            "DE_population_size": 8.0,
            "scaling_factor": 0.9,
            "binomial_crossover_probability": 0.87,
            "PSO_population_size": 89,
            "inertia": 0.76,
            "cognitive_update_rate": 0.87,
            "social_update_rate": 1.18
        }
    }
}


def main():

    classification = [
        "Project 3/data/glass.data",
        "Project 3/data/breast-cancer-wisconsin.data",
        "Project 3/data/soybean-small.data"
    ]

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for file in classification:
        data = RootData(file, True)
        filename = os.path.splitext(os.path.basename(file))[0]

        for num_hidden_layers in range(3):

            ## HYPERPARAMETERS ##
            GA_population_size = tuned_hyperparameters_classification[filename][num_hidden_layers]["GA_population_size"]
            GA_crossover_rate =  tuned_hyperparameters_classification[filename][num_hidden_layers]["crossover_rate"]
            GA_mutation_rate = tuned_hyperparameters_classification[filename][num_hidden_layers]["mutation_rate"]
           
            GA_hyperparameters =   {
            'population_size': int(GA_population_size),
            'crossover_rate': GA_crossover_rate,
            'mutation_rate': GA_mutation_rate,
            'num_hidden_nodes': num_hidden_layers,
            'learning_rate': 0.01,
            }

            DE_population_size = tuned_hyperparameters_classification[filename][num_hidden_layers]["DE_population_size"]
            DE_scaling_factor = tuned_hyperparameters_classification[filename][num_hidden_layers]["scaling_factor"]
            DE_binomial_crossover_probability = tuned_hyperparameters_classification[filename][num_hidden_layers]["binomial_crossover_probability"]

            DE_hyperparameters = {
            'population_size': int(DE_population_size),
            'scaling_factor': DE_scaling_factor,
            'binomial_crossover_probability': DE_binomial_crossover_probability,
            'num_hidden_nodes': num_hidden_layers,
            'learning_rate': 0.01,
            }   

            PSO_population_size = tuned_hyperparameters_classification[filename][num_hidden_layers]["PSO_population_size"]
            PSO_inertia = tuned_hyperparameters_classification[filename][num_hidden_layers]["inertia"]
            PSO_cognitive_update_rate = tuned_hyperparameters_classification[filename][num_hidden_layers]["cognitive_update_rate"]
            PSO_social_update_rate = tuned_hyperparameters_classification[filename][num_hidden_layers]["social_update_rate"]


            PSO_hyperparameters = {
            'population_size': PSO_population_size, 
            'inertia': PSO_inertia, 
            'cognitive_update_rate': PSO_cognitive_update_rate, 
            'social_update_rate': PSO_social_update_rate, 
            'num_hidden_nodes':num_hidden_layers}

            ###

            all_predictions_DE = []
            all_actual_DE = []

            all_predictions_GA = []
            all_actual_GA = []

            all_predictions_PSO = []
            all_actual_PSO = []

            # Perform cross-validation for each fold
            for fold in folds:
                print(f"Started with fold: {fold}, hidden layers: {num_hidden_layers}, dataset: {filename}")

                de = DifferentialEvolution(data=data, hold_out_fold=fold, number_hidden_layers=num_hidden_layers, hyperparameters=DE_hyperparameters)
                de.train()
                prediction, actual = de.test()
                all_predictions_DE.extend(prediction)
                all_actual_DE.extend(actual)
                print("DA done")

                ga = GeneticAlgorithm(data=data, hold_out_fold=fold, number_hidden_layers=num_hidden_layers, hyperparameters=GA_hyperparameters)
                ga.train()
                prediction, actual = ga.test()
                all_predictions_GA.extend(prediction)
                all_actual_GA.extend(actual)
                print("GA done")

                pso = PSO(data=data, hold_out_fold=fold, number_hidden_layers=num_hidden_layers, hyperparameters=PSO_hyperparameters)
                pso.train()
                prediction, actual = pso.test()
                all_predictions_PSO.extend(prediction)
                all_actual_PSO.extend(actual)
                print("PSO done")

                print(f"Done with fold {fold} : {filename}")

            ConfusionMatrix(all_actual_DE, all_predictions_DE).print_confusion_matrix()
            ConfusionMatrix(all_actual_GA, all_predictions_GA).print_confusion_matrix()
            ConfusionMatrix(all_actual_PSO, all_predictions_PSO).print_confusion_matrix()

main()