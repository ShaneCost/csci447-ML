import random
from genetic_algorthim import *
from differential_evolution import *
from PSO import *
from root_data import *
from hyperparameter import *
from loss import *

def tune_ga(data, layer):
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    population_size = Hyperparameter("population_size", 5, 100)
    crossover_rate = Hyperparameter("crossover_rate", 0, 1)
    mutation_rate = Hyperparameter("mutation_rate", 0, 1)

    # Begin tuning population size
    print("\ttuning pop size")
    count = 0
    while not population_size.is_tuned:
        hyperparameters = { # Dictionary for all hyperparameters
            'population_size': round(population_size.value),
            'crossover_rate': .5,
            'mutation_rate': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0 # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = GeneticAlgorithm(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        population_size.update(score)
        count += 1
        print("\t\tupdated ", count)

    # Begin tuning crossover size
    print("\ttuning crossover rate")
    count = 0
    while not crossover_rate.is_tuned:
        hyperparameters = { # Dictionary for all hyperparameters
            'population_size': 20,
            'crossover_rate': crossover_rate.value,
            'mutation_rate': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0 # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = GeneticAlgorithm(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        crossover_rate.update(score)
        count += 1
        print("\t\tupdated ", count)

    # Begin tuning mutation rate
    print("\ttuning mutation rate")
    count = 0
    while not mutation_rate.is_tuned:
        hyperparameters = { # Dictionary for all hyperparameters
            'population_size': 20,
            'crossover_rate': .5,
            'mutation_rate': mutation_rate.value,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0 # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = GeneticAlgorithm(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        mutation_rate.update(score)
        count += 1
        print("\t\tupdated ", count)

    return population_size.value, crossover_rate.value, mutation_rate.value


def tune_de(data, layer):
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    population_size = Hyperparameter("population_size", 5, 100)
    scaling_factor = Hyperparameter("scaling_factor", 0, 1)
    binomial_crossover_probability = Hyperparameter("binomial_crossover_probability", 0, 1)

    # Begin tuning population size
    print("\ttuning pop size")
    count = 0
    while not population_size.is_tuned:
        hyperparameters = { # Dictionary for all hyperparameters
            'population_size': round(population_size.value),
            'scaling_factor': .5,
            'binomial_crossover_probability': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0 # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = DifferentialEvolution(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        population_size.update(score)
        count += 1
        print("\t\tupdated ", count)

    # Begin tuning scaling factor
    print("\ttuning scaling factor")
    count = 0
    while not scaling_factor.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': 20,
            'scaling_factor': scaling_factor.value,
            'binomial_crossover_probability': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = DifferentialEvolution(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        scaling_factor.update(score)
        count += 1
        print("\t\tupdated, ", count)

    # Begin tuning binomial crossover probability
    print("\ttuning binomial crossover")
    count = 0
    while not binomial_crossover_probability.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': 20,
            'scaling_factor': .5,
            'binomial_crossover_probability': binomial_crossover_probability.value,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = DifferentialEvolution(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        binomial_crossover_probability.update(score)
        count += 1
        print("\t\tupdated, ", count)

    return population_size.value, scaling_factor.value, binomial_crossover_probability.value

def tune_pso(data, layer):
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    population_size = Hyperparameter("population_size", 5, 100)
    inertia = Hyperparameter("inertia", 0.1, 1.5)
    cognitive_update_rate = Hyperparameter("cognitive_update_rate", 0.4, 2)
    social_update_rate = Hyperparameter("social_update_rate", 0.4, 2)

    # Begin tuning population size
    print("\ttuning pop size")
    count = 0
    while not population_size.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': round(population_size.value),
            'inertia': .5,
            'cognitive_update_rate': .5,
            'social_update_rate': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = PSO(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        population_size.update(score)
        count += 1
        print("\t\tupdated, ", count)

    # Begin tuning inertia
    print("\ttuning inertia")
    count = 0
    while not inertia.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': 20,
            'inertia': inertia.value,
            'cognitive_update_rate': .5,
            'social_update_rate': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = PSO(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        inertia.update(score)
        count += 1
        print("\t\tupdated, ", count)

    # Begin tuning cognitive update rate
    print("\ttuning cognitive update_rate")
    count = 0
    while not cognitive_update_rate.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': 20,
            'inertia': .5,
            'cognitive_update_rate': cognitive_update_rate.value,
            'social_update_rate': .5,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = PSO(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        cognitive_update_rate.update(score)
        count += 1
        print("\t\tupdated, ", count)

    # Begin tuning social update rate
    print("\ttuning social update_rate")
    while not social_update_rate.is_tuned:
        hyperparameters = {  # Dictionary for all hyperparameters
            'population_size': 20,
            'inertia': .5,
            'cognitive_update_rate': .5,
            'social_update_rate': social_update_rate.value,
            'num_hidden_nodes': 25,
            'learning_rate': 0.014
        }
        score = 0  # Variable used to accumulate performance
        # Iterate over the folds
        for fold in folds:
            ga = PSO(data=data, hold_out_fold=fold, number_hidden_layers=layer, hyperparameters=hyperparameters)
            print("\t\t\t\ttraining ", fold)
            ga.train()
            prediction, actual = ga.test()
            loss = Loss(prediction, actual, is_class=data.is_class)
            score += loss.results
        score /= 10
        social_update_rate.update(score)
        count += 1
        print("\t\tupdated, ", count)

    return population_size.value, inertia.value, cognitive_update_rate.value, social_update_rate.value

def main():
    num_hidden_layers = [0, 1, 2]
    # All data set used for classification problems
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]

    # All data set used for regression problems
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data"]

    with open("tuned_values.txt", "w") as f:  # Open a file for writing
        for file in classification:
            data = RootData(file, is_class=True)
            print("\n", data.name)
            print("\n", data.name, file=f)  # Write to the file
            for layer in num_hidden_layers:
                population_size, crossover_rate, mutation_rate = tune_ga(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tGA population size: ', population_size)
                print('\t\tcrossover rate: ', crossover_rate)
                print('\t\tmutation rate: ', mutation_rate)
                print('\t', layer, " hidden layers", file=f)
                print('\n\t\tGA population size: ', population_size, file=f)
                print('\t\tcrossover rate: ', crossover_rate, file=f)
                print('\t\tmutation rate: ', mutation_rate, file=f)

                population_size, scaling_factor, binomial_crossover_probability = tune_de(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tDE population size: ', population_size)
                print('\t\tscaling factor: ', scaling_factor)
                print('\t\tbinomial crossover probability: ', binomial_crossover_probability)
                print('\n\t\tDE population size: ', population_size, file=f)
                print('\t\tscaling factor: ', scaling_factor, file=f)
                print('\t\tbinomial crossover probability: ', binomial_crossover_probability, file=f)

                population_size, inertia, cognitive_update_rate, social_update_rate = tune_pso(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tPSO population size: ', population_size)
                print('\t\tinertia: ', inertia)
                print('\t\tcognitive update rate: ', cognitive_update_rate)
                print('\t\tsocial update rate: ', social_update_rate)
                print('\n\t\tPSO population size: ', population_size, file=f)
                print('\t\tinertia: ', inertia, file=f)
                print('\t\tcognitive update rate: ', cognitive_update_rate, file=f)
                print('\t\tsocial update rate: ', social_update_rate, file=f)

        for file in regression:
            data = RootData(file, is_class=False)
            print("\n", data.name)
            print("\n", data.name, file=f)
            for layer in num_hidden_layers:
                population_size, crossover_rate, mutation_rate = tune_ga(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tGA population size: ', population_size)
                print('\t\tcrossover rate: ', crossover_rate)
                print('\t\tmutation rate: ', mutation_rate)
                print('\t', layer, " hidden layers", file=f)
                print('\t\tGA population size: ', population_size, file=f)
                print('\t\tcrossover rate: ', crossover_rate, file=f)
                print('\t\tmutation rate: ', mutation_rate, file=f)

                population_size, scaling_factor, binomial_crossover_probability = tune_de(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tDE population size: ', population_size)
                print('\t\tscaling factor: ', scaling_factor)
                print('\t\tbinomial crossover probability: ', binomial_crossover_probability)
                print('\t\tDE population size: ', population_size, file=f)
                print('\t\tscaling factor: ', scaling_factor, file=f)
                print('\t\tbinomial crossover probability: ', binomial_crossover_probability, file=f)

                population_size, inertia, cognitive_update_rate, social_update_rate = tune_pso(data, layer)
                print("\n", data.name)
                print('\t', layer, " hidden layers")
                print('\n\t\tPSO population size: ', population_size)
                print('\t\tinertia: ', inertia)
                print('\t\tcognitive update rate: ', cognitive_update_rate)
                print('\t\tsocial update rate: ', social_update_rate)
                print('\t\tPSO population size: ', population_size, file=f)
                print('\t\tinertia: ', inertia, file=f)
                print('\t\tcognitive update rate: ', cognitive_update_rate, file=f)
                print('\t\tsocial update rate: ', social_update_rate, file=f)
main()
