from feedforward_neural_network import *
from meta_data import *
from root_data import *
from loss import *
from PSO import *
from genetic_algorthim import *
from differential_evolution import *
import os

tuned_hyperparameters_regression = {
    "forestfires": {
        0 : {
            "GA_population_size": 15.0,
            "crossover_rate": 0.53,
            "mutation_rate": 0.0,
            "DE_population_size": 71.0,
            "scaling_factor": 0.01,
            "binomial_crossover_probability": 0.5,
            "PSO_population_size": 10,
            "inertia": 0.13,
            "cognitive_update_rate": 0.78,
            "social_update_rate": 0.46
        },
        1 : {
            "GA_population_size": 5.0,
            "crossover_rate": 0.05,
            "mutation_rate": 0.03,
            "DE_population_size": 18,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 0.33,
            "PSO_population_size": 27,
            "inertia": 0.89,
            "cognitive_update_rate": 0.36,
            "social_update_rate": 1.14
        },
        2 : {
            "GA_population_size": 8.0,
            "crossover_rate": 0.10,
            "mutation_rate": 0.14,
            "DE_population_size": 6.0,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 0.95,
            "PSO_population_size": 7,
            "inertia": 0.31,
            "cognitive_update_rate": 0.18,
            "social_update_rate": 0.64
        }
    },
    "machine": {
        0 : {
            "GA_population_size": 18.0,
            "crossover_rate": 0.76,
            "mutation_rate": 0.09,
            "DE_population_size": 99,
            "scaling_factor": 0.80,
            "binomial_crossover_probability": 0.50,
            "PSO_population_size": 5.0,
            "inertia": 0.4,
            "cognitive_update_rate": 0.94,
            "social_update_rate": 0.31
        },
        1 : {
            "GA_population_size": 4.0,
            "crossover_rate": 0.56,
            "mutation_rate": 0.61,
            "DE_population_size": 24,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 1.0,
            "PSO_population_size": 67,
            "inertia": 0.72,
            "cognitive_update_rate": 0.53,
            "social_update_rate": 1.19
        },
        2 : {
            "GA_population_size": 11.0,
            "crossover_rate": 0.00,
            "mutation_rate": 0.00,
            "DE_population_size": 59.0,
            "scaling_factor": 1.0,
            "binomial_crossover_probability": 0.31,
            "PSO_population_size": 91,
            "inertia": 0.18,
            "cognitive_update_rate": 0.73,
            "social_update_rate": 1.90
        }
    },
    "abalone": {
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
    }
}



def main():

    classification = [
        "Project 3/data/abalone.data"
    ]

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for file in classification:
        data = RootData(file, False)
        filename = os.path.splitext(os.path.basename(file))[0]

        for num_hidden_layers in range(3):

            ## HYPERPARAMETERS ##
            GA_population_size = tuned_hyperparameters_regression[filename][num_hidden_layers]["GA_population_size"]
            GA_crossover_rate =  tuned_hyperparameters_regression[filename][num_hidden_layers]["crossover_rate"]
            GA_mutation_rate = tuned_hyperparameters_regression[filename][num_hidden_layers]["mutation_rate"]
           
            GA_hyperparameters =   {
            'population_size': int(GA_population_size),
            'crossover_rate': GA_crossover_rate,
            'mutation_rate': GA_mutation_rate,
            'num_hidden_nodes': num_hidden_layers,
            'learning_rate': 0.01,
            }

            DE_population_size = tuned_hyperparameters_regression[filename][num_hidden_layers]["DE_population_size"]
            DE_scaling_factor = tuned_hyperparameters_regression[filename][num_hidden_layers]["scaling_factor"]
            DE_binomial_crossover_probability = tuned_hyperparameters_regression[filename][num_hidden_layers]["binomial_crossover_probability"]

            DE_hyperparameters = {
            'population_size': int(DE_population_size),
            'scaling_factor': DE_scaling_factor,
            'binomial_crossover_probability': DE_binomial_crossover_probability,
            'num_hidden_nodes': num_hidden_layers,
            'learning_rate': 0.01,
            }   

            PSO_population_size = tuned_hyperparameters_regression[filename][num_hidden_layers]["PSO_population_size"]
            PSO_inertia = tuned_hyperparameters_regression[filename][num_hidden_layers]["inertia"]
            PSO_cognitive_update_rate = tuned_hyperparameters_regression[filename][num_hidden_layers]["cognitive_update_rate"]
            PSO_social_update_rate = tuned_hyperparameters_regression[filename][num_hidden_layers]["social_update_rate"]


            PSO_hyperparameters = {
            'population_size': int(PSO_population_size), 
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
            # for fold in folds:
            #     print(f"Started with fold: {fold}, hidden layers: {num_hidden_layers}, dataset: {filename}")

            de = DifferentialEvolution(data=data, hold_out_fold=1, number_hidden_layers=num_hidden_layers, hyperparameters=DE_hyperparameters)
            de.train()
            prediction, actual = de.test()
            all_predictions_DE.extend(prediction)
            all_actual_DE.extend(actual)
            print("DA done")

            ga = GeneticAlgorithm(data=data, hold_out_fold=1, number_hidden_layers=num_hidden_layers, hyperparameters=GA_hyperparameters)
            ga.train()
            prediction, actual = ga.test()
            all_predictions_GA.extend(prediction)
            all_actual_GA.extend(actual)
            print("GA done")

            pso = PSO(data=data, hold_out_fold=1, number_hidden_layers=num_hidden_layers, hyperparameters=PSO_hyperparameters)
            pso.train()
            prediction, actual = pso.test()
            all_predictions_PSO.extend(prediction)
            all_actual_PSO.extend(actual)
            print("PSO done")

                # print(f"Done with fold {fold} : {filename}")

            DE_loss =  Loss(all_predictions_DE, all_actual_DE, False).mean_squared_error()
            GA_loss = Loss(all_predictions_GA, all_actual_GA, False).mean_squared_error()
            PSO_loss = Loss(all_predictions_PSO, all_actual_PSO, False).mean_squared_error()

            print(f"DE loss {DE_loss}") 
            print(f"DE loss {GA_loss}") 
            print(f"DE loss {PSO_loss}") 

main()