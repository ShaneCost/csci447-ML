__author__ = "<Shane Costello>"

from root_data import *
from genetic_algorthim import *
from differential_evolution import *
from PSO import *
from feedforward_neural_network import *
from loss import *

def requirement_1(classification, regression, hyperparameters):
    """"
    REQUIREMENT 1
    Provide sample outputs from one test fold showing performance on one classification and one regression
    network. Show results for the two hidden layer cases only but for each of the learning methods.
    """""

    print("CLASSIFICATION")

    print("\nGenetic Algorithm")
    ga = GeneticAlgorithm(data=classification,hold_out_fold=1,number_hidden_layers=2,hyperparameters=hyperparameters)
    ga.train()
    predict, actual = ga.test()
    loss = Loss(predict, actual)
    loss.confusion_matrix.print_confusion_matrix()

    print("\nDifferential Evolution")
    de = DifferentialEvolution(data=classification,hold_out_fold=1,number_hidden_layers=2,hyperparameters=hyperparameters)
    de.train()
    predict, actual = de.test()
    loss = Loss(predict, actual)
    loss.confusion_matrix.print_confusion_matrix()

    print("\nParticle Swarm Optimization")
    pso = PSO(data=classification,hold_out_fold=1,number_hidden_layers=2,hyperparameters=hyperparameters)
    pso.train()
    predict, actual = pso.test()
    loss = Loss(predict, actual)
    loss.confusion_matrix.print_confusion_matrix()

    print("\nBackpropagation")
    backprop = FeedForwardNetwork(data=classification,hold_out_fold=1,number_hidden_layers=2,hyperparameters=hyperparameters, _id=0)
    backprop.train()
    predict, actual = backprop.test()
    loss = Loss(predict, actual)
    loss.confusion_matrix.print_confusion_matrix()

    print("\nREGRESSION")

    print("\nGenetic Algorithm")
    ga = GeneticAlgorithm(data=regression, hold_out_fold=1, number_hidden_layers=2, hyperparameters=hyperparameters)
    ga.train()
    predict, actual = ga.test()
    loss = Loss(predict, actual, is_class=False)
    print("mse: ", loss.results)

    print("\nDifferential Evolution")
    de = DifferentialEvolution(data=regression, hold_out_fold=1, number_hidden_layers=2,hyperparameters=hyperparameters)
    de.train()
    predict, actual = de.test()
    loss = Loss(predict, actual, is_class=False)
    print("mse: ", loss.results)

    print("\nParticle Swarm Optimization")
    pso = PSO(data=regression, hold_out_fold=1, number_hidden_layers=2, hyperparameters=hyperparameters)
    pso.train()
    predict, actual = pso.test()
    loss = Loss(predict, actual, is_class=False)
    print("mse: ", loss.results)

    print("\nBackpropagation")
    backprop = FeedForwardNetwork(data=regression, hold_out_fold=1, number_hidden_layers=2, hyperparameters=hyperparameters, _id=0)
    backprop.train()
    predict, actual = backprop.test()
    loss = Loss(predict, actual, is_class=False)
    print("mse: ", loss.results)

def requirement_2(data, hyperparameters):
    """
    REQUIREMENT 2
    Demonstrate each of the main operations for the GA: selection, crossover, and mutation.
    """""
    print("\nGENETIC ALGORTHM")

    ga = GeneticAlgorithm(data=data, hold_out_fold=2, number_hidden_layers=1, hyperparameters=hyperparameters, demo=True)

    # SELECTION
    ga.fitness_proportionate_selection()

    # CROSSOVER
    ga.arithmetic_crossover()

    # MUTATION
    ga.order_based_mutation()

def requirement_3(data, hyperparameters):
    """"
    REQUIREMENT 3
    Demonstrate each of the main operations for the DE: crossover and mutation.
    """""
    print("\nDIFFERENTIAL EVOLUTION")

    de = DifferentialEvolution(data=data, hold_out_fold=2, number_hidden_layers=2, hyperparameters=hyperparameters, demo=True)

    # MUTATION
    de.selection()
    de.create_mutant()

    # CROSSOVER
    de.crossover()

def requirement_4():
    """"
    REQUIREMENT 4
    Demonstrate each of the main operations for the PSO: pbest calculation, gbest calculation, velocity
    update, and position update.
    """""

    # TODO: complete requirement_4()

def requirement_5(classification, regression, hyperparameters):
    """"
    REQUIREMENT 5
    Show the average performance over the ten folds for one of the classification data sets and one of the
    regression data sets for each of the networks trained with each of the algorithms
    """""
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(classification.name)
    for size in range(3):
        print(f"\n{size} Hidden Layers")

        predictions = []
        actuals = []
        for fold in folds:
            ga = GeneticAlgorithm(data=classification, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals)
        print("\nGenetic Algorithm")
        loss.confusion_matrix.print_confusion_matrix()

        predictions = []
        actuals = []
        for fold in folds:
            ga = DifferentialEvolution(data=classification, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals)
        print("\nDifferential Evolution")
        loss.confusion_matrix.print_confusion_matrix()

        predictions = []
        actuals = []
        for fold in folds:
            ga = PSO(data=classification, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals)
        print("\nParticle Swarm Optimization")
        loss.confusion_matrix.print_confusion_matrix()

        predictions = []
        actuals = []
        for fold in folds:
            ga = FeedForwardNetwork(data=classification, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters, _id=0)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals)
        print("\nBackpropagation")
        loss.confusion_matrix.print_confusion_matrix()

    print(f"\n{regression.name}")
    for size in range(3):
        print(f"\n{size} Hidden Layers")

        predictions = []
        actuals = []
        for fold in folds:
            ga = GeneticAlgorithm(data=regression, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals, is_class=False)
        print("\nGenetic Algorithm")
        print("mse: ", loss.results)

        predictions = []
        actuals = []
        for fold in folds:
            ga = DifferentialEvolution(data=regression, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals, is_class=False)
        print("\nDifferential Evolution")
        print("mse: ", loss.results)

        predictions = []
        actuals = []
        for fold in folds:
            ga = PSO(data=regression, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals, is_class=False)
        print("\nParticle Swarm Optimization")
        print("mse: ", loss.results)

        predictions = []
        actuals = []
        for fold in folds:
            ga = FeedForwardNetwork(data=regression, number_hidden_layers=size, hold_out_fold=fold, hyperparameters=hyperparameters, _id=0)
            ga.train()
            predict, actual = ga.test()
            predictions.extend(predict)
            actuals.extend(actual)
        loss = Loss(predictions, actuals, is_class=False)
        print("\nBackpropagation")
        print("mse: ", loss.results)

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

    classification = RootData("../data/soybean-small.data", is_class=True)
    regression = RootData("../data/forestfires.data", is_class=False)

    input('\nStart\n')
    # requirement_1(classification, regression, hyperparameters)

    input('\nContinue\n')
    # requirement_2(classification, hyperparameters)

    input('\nContinue\n')
    # requirement_3(classification, hyperparameters)

    input('\nContinue\n')
    # requirement_4()

    input('\nContinue\n')
    requirement_5(classification, regression, hyperparameters)

if __name__ == '__main__':
    main()