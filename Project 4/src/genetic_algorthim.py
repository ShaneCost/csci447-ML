__author__ = "<Shane Costello>"

from feedforward_neural_network import *
import random
import math

class GeneticAlgorithm:
    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters, demo=False):
        """
        Class used to define the Genetic Algorithm

        :param data: instance of our RootData class
        :param hold_out_fold: integer indicating the current hold out fold our experiment
        :param number_hidden_layers: the number of hidden layers in each neural network
        :param hyperparameters: a dictionary containing all relevant hyperparameters
        """

        self.demo = demo

        # Initialize hyperparameters
        self.population_size = hyperparameters['population_size']
        self.crossover_rate = hyperparameters['crossover_rate']
        self.mutation_rate = hyperparameters['mutation_rate']

        # Initialize data attributes
        self.data = data
        self.hold_out_fold = hold_out_fold
        self.number_hidden_layers = number_hidden_layers
        self.hyperparameters = hyperparameters

        # Placeholders and variables
        self.population = []
        self.current_id = 0
        self.current_parent_1 = None
        self.current_parent_2 = None
        self.current_child = None
        self.converged = False
        self.generation = 1

        self.initialize_population()


    def initialize_population(self):
        """
        Method used to create the initial population
        """

        _id = 0
        for i in range(self.population_size):
            neural_network = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, _id)
            _id += 1
            self.population.append(neural_network)
        self.current_id = _id + 1

        # Derive initial fitness value
        for member in self.population:
            member.forward_pass()

    def fitness_proportionate_selection(self):
        """
        Method used to perform the fitness proportionate selection
        """

        # Calculate total fitness
        total_fitness = sum(member.fitness for member in self.population)

        # Normalize fitness values to create probabilities
        if total_fitness == 0:
            weights = [1 / len(self.population)] * len(self.population)  # Uniform probabilities
        else:
            weights = [member.fitness / total_fitness for member in self.population]

        # Use random to select two parents
        parents = random.choices(self.population, weights=weights, k=2)

        self.current_parent_1 = parents[0]
        self.current_parent_2 = parents[1]

        if self.demo:
            print("\nFitness Proportionate Selection")
            print("\tPopulation's selection probability")
            for member in self.population:
                print(f"\t\tMember {member.id}: {member.fitness}/{total_fitness} = {member.fitness/total_fitness}")
            print(f"\n\tParent 1 = Member {self.current_parent_1.id}")
            print(f"\tParent 2 = Member {self.current_parent_2.id}")

    def arithmetic_crossover(self):
        """
        Method used to perform the arithmetic crossover
        """
        # Make child
        child = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, self.current_id)
        self.current_id += 1

        if self.demo:
            print("\nArithmetic Crossover")
            print("\tWeight crossover")

        # Edge crossover
        for parent_1_edge, parent_2_edge, child_edge in zip(self.current_parent_1.edge_set.edges, self.current_parent_2.edge_set.edges, child.edge_set.edges):
            child_edge.weight = ((parent_1_edge.weight + parent_2_edge.weight) / 2)

            if self.demo:
                print(f"\t\tchild weight = 0.5({parent_1_edge.weight} + {parent_2_edge.weight})")
                print(f"\t\t{child_edge.weight}\n")

        # Input layer bias
        if self.demo:
            print("\tBias crossover")

        for parent_1_node, parent_2_node, child_node in zip(self.current_parent_1.node_set.input_layer, self.current_parent_2.node_set.input_layer, child.node_set.input_layer):
            child_node.bias = ((parent_1_node.bias + parent_2_node.bias) / 2)

            if self.demo:
                print(f"\t\tchild bias = 0.5({parent_1_node.bias} + {parent_2_node.bias})")
                print(f"\t\t{child_node.bias}\n")

        # Hidden layers bias
        for p_1_layer, p_2_layer, c_later in zip(self.current_parent_1.node_set.hidden_layers, self.current_parent_2.node_set.hidden_layers, child.node_set.hidden_layers):
            for parent_1_node, parent_2_node, child_node in zip(p_1_layer, p_2_layer, c_later):
                child_node.bias = ((parent_1_node.bias + parent_2_node.bias) / 2)

        # Output layer bias
        for parent_1_node, parent_2_node, child_node in zip(self.current_parent_1.node_set.output_layer, self.current_parent_2.node_set.output_layer, child.node_set.output_layer):
            child_node.bias = ((parent_1_node.bias + parent_2_node.bias) / 2)

        self.current_child = child

    def order_based_mutation(self):
        """
        Method used to perform the order based mutation
        """
        number_of_edge_mutations = math.ceil(len(self.current_child.edge_set.edges) * self.mutation_rate) # Calculate the number of mutations to perform

        if self.demo:
            print("\nOrder based Mutation")
            print("\tWeight Mutation")
            print(f"\tNumber of edge mutations: {len(self.current_child.edge_set.edges)} * {self.mutation_rate} = {number_of_edge_mutations}")

        max_v = len(self.current_child.edge_set.edges) - 1
        for i in range(number_of_edge_mutations):
            # Randomly select two edges in the child
            edge_1 = self.current_child.edge_set.edges[random.randint(0, max_v)]
            edge_2 = self.current_child.edge_set.edges[random.randint(0, max_v)]

            # Mutate
            edge_1_weight = edge_1.weight
            edge_1.weight = edge_2.weight
            edge_2.weight = edge_1_weight

            if self.demo:
                print(f"\t\tMutation {i}: Swapping {edge_1.weight} and {edge_2.weight}")

        num_nodes = len(self.current_child.node_set.input_layer) + (self.number_hidden_layers * self.hyperparameters['num_hidden_nodes']) + len(self.current_child.node_set.output_layer)
        number_of_node_mutations = math.ceil(num_nodes * self.mutation_rate)

        for i in range(number_of_node_mutations):
            layer = random.randint(0, 2)
            if layer == 0:
                max_v = len(self.current_child.node_set.input_layer) - 1

                node_1 = self.current_child.node_set.input_layer[random.randint(0, max_v)]
                node_2 = self.current_child.node_set.input_layer[random.randint(0, max_v)]
            elif layer == 1:
                if self.number_hidden_layers == 0:
                    node_1 = self.current_child.node_set.input_layer[random.randint(0, 4)]
                    node_2 = self.current_child.node_set.input_layer[random.randint(0, 4)]
                else:
                    h_layer = random.randint(0, self.number_hidden_layers - 1)

                    max_v = len(self.current_child.node_set.hidden_layers[h_layer]) - 1
                    node_1 = self.current_child.node_set.hidden_layers[h_layer][random.randint(0, max_v)]
                    node_2 = self.current_child.node_set.hidden_layers[h_layer][random.randint(0, max_v)]
            else:
                max_v = len(self.current_child.node_set.output_layer) - 1

                node_1 = self.current_child.node_set.output_layer[random.randint(0, max_v)]
                node_2 = self.current_child.node_set.output_layer[random.randint(0, max_v)]

            node_1_bias = node_1.bias
            node_1.bias = node_2.bias
            node_2.bias = node_1_bias

    def read_state_replacement(self):
        """
        Method to implement reading state replacement
        """
        # Find the two least fit members
        sorted_population = sorted(self.population, key=lambda member: member.fitness)
        least_fit_1 = sorted_population[0]
        least_fit_2 = sorted_population[1]

        # Replace the two least fit members with the new child
        self.population.remove(least_fit_1)
        self.population.remove(least_fit_2)

        # Add the new child to the population
        self.population.append(self.current_child)
        self.population.append(self.current_child)
        self.current_child.forward_pass() # Derive child's fitness value

        self.current_parent_1 = None
        self.current_parent_2 = None
        self.current_child = None

    def check_convergence(self):
        """
        Method used to check the convergence of the algorithm
        """
        self.generation += 1

        if self.generation < (self.population_size/2):
            pass
        else:
            if self.generation >= 3000:
                self.converged = True

    def log_progress(self):
        best_fitness = max(member.fitness for member in self.population)
        avg_fitness = sum(member.fitness for member in self.population) / len(self.population)
        # print(f"Generation {self.generation}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")
        return avg_fitness

    def train(self):
        fitness_values = []
        while not self.converged:
            self.fitness_proportionate_selection()
            self.arithmetic_crossover()
            self.order_based_mutation()
            self.read_state_replacement()
            self.check_convergence()
            fitness_values.append(self.log_progress())

        return fitness_values

    def test(self):
        sorted_population = sorted(self.population, key=lambda member: member.fitness, reverse=True)
        most_fit = sorted_population[0]
        prediction, actual = most_fit.test()

        return prediction, actual

# from root_data import *
# def main():
#     classification = "../data/soybean-small.data"
#     regression = "../data/forestfires.data"

#     classification_data = RootData(path=classification, is_class=True)
#     regression_data = RootData(path=regression, is_class=False)

#     hyperparameters = {
#         'population_size': 10,
#         'crossover_rate': 0.8,
#         'mutation_rate': 0.1,
#         'num_hidden_nodes': 2,
#         'learning_rate': 0.01,
#     }

#     classification_ga = GeneticAlgorithm(data=classification_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
#     classification_ga.train()
#     prediction, actual = classification_ga.test()

#     regression_ga = GeneticAlgorithm(data=regression_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
#     regression_ga.train()
#     prediction, actual = regression_ga.test()

# main()