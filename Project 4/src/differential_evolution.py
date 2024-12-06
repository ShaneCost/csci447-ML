__author__ = "<Shane Costello>"

from feedforward_neural_network import *
import random
from node import *

class DifferentialEvolution:

    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters, demo=False):
        """
        Class used to model the differential evolution problem.

        :param data: instance of our RootData class
        :param hold_out_fold: number indicating the current holdout fold for our experiment
        :param number_hidden_layers: the number of hidden layers in each member of the population
        :param hyperparameters: dictionary containing the hyperparameters of the model
        """
        self.demo = demo

        # Initialize hyperparameters
        self.population_size = hyperparameters['population_size']
        self.scaling_factor = hyperparameters['scaling_factor']
        self.binomial_crossover_probability = hyperparameters['binomial_crossover_probability']

        # Data and model configuration
        self.data = data
        self.hold_out_fold = hold_out_fold
        self.number_hidden_layers = number_hidden_layers
        self.hyperparameters = hyperparameters

        # Variables used to store state of the model
        self.population = []
        self.generation = 1
        self.current_id = 0
        self.converged = False

        # Placeholders
        self.target_member = None
        self.a = None
        self.b = None
        self.c = None
        self.mutant = None
        self.child = None

        # Make the population
        self.initialize_population()

    def initialize_population(self):
        """"
        Method used to initialize the population.
        """

        # Create all members of the population
        _id = 0
        for i in range(self.population_size):
            neural_network = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, _id)
            _id += 1
            self.population.append(neural_network)
        self.current_id = _id + 1

        # Derive initial fitness value
        for member in self.population:
            member.forward_pass()

    def selection(self):
        """
        Method used to select three unique members of the population.
        """
        if self.demo:
            self.target_member = self.population[0]

        # Get target member ID
        target_member_id = self.target_member.id

        # Create a range of IDs excluding the target member ID
        valid_ids = [i for i in range(self.population_size) if i != target_member_id]

        # Randomly sample 3 unique IDs from the valid IDs
        ids = random.sample(valid_ids, 3)

        # Get neural networks
        self.a = self.get_member(ids[0])
        self.b = self.get_member(ids[1])
        self.c = self.get_member(ids[2])

        if self.demo:
            print(f"\nTarget Member ID: {target_member_id}")
            print(f"\tRandomly selected population members...")
            print(f"\t\tMember {ids[0]}")
            print(f"\t\tMember {ids[1]}")
            print(f"\t\tMember {ids[2]}")

    def get_member(self, _id):
        """
        Method used to return a member of the population.

        :param _id: integer
        :return: instance of the FeedForwardNetwork class
        """
        for member in self.population:
            if member.id == _id:
                return member

    def create_mutant(self):
        """"
        Method used to create the mutant vectors
        """
        if self.demo:
            print("\n")
        # Unroll nodes and edges for a, b, c
        a_unrolled_nodes = self.a.node_set
        b_unrolled_nodes = self.b.node_set
        c_unrolled_nodes = self.c.node_set

        a_unrolled_edges = self.a.edge_set.edges
        b_unrolled_edges = self.b.edge_set.edges
        c_unrolled_edges = self.c.edge_set.edges

        # Create new NodeSet
        new_node_set = NodeSet()

        # Map original nodes to new nodes
        node_mapping = {}

        if self.demo:
            print('\nMutant Vector')

        # Input layer
        input_layer = []
        for a_node, b_node, c_node in zip(a_unrolled_nodes.input_layer, b_unrolled_nodes.input_layer, c_unrolled_nodes.input_layer):
            difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
            if self.demo:
                print(f"(({b_node.bias} - {c_node.bias}) * {self.scaling_factor}) + {a_node.bias} = {difference}")
            new_node = Node(difference)
            input_layer.append(new_node)
            node_mapping[a_node] = new_node
        new_node_set.input_layer = input_layer

        # Hidden layers
        hidden_layers = []
        for a_layer, b_layer, c_layer in zip(a_unrolled_nodes.hidden_layers, b_unrolled_nodes.hidden_layers, c_unrolled_nodes.hidden_layers):
            layer = []
            for a_node, b_node, c_node in zip(a_layer, b_layer, c_layer):
                difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
                if self.demo:
                    print(f"(({b_node.bias} - {c_node.bias}) * {self.scaling_factor}) + {a_node.bias} = {difference}")
                new_node = Node(difference)
                layer.append(new_node)
                node_mapping[a_node] = new_node
            hidden_layers.append(layer)
        new_node_set.hidden_layers = hidden_layers

        # Output layer
        output_layer = []
        for a_node, b_node, c_node in zip(a_unrolled_nodes.output_layer, b_unrolled_nodes.output_layer, c_unrolled_nodes.output_layer):
            difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
            if self.demo:
                print(f"(({b_node.bias} - {c_node.bias}) * {self.scaling_factor}) + {a_node.bias} = {difference}")
            new_node = Node(difference)
            output_layer.append(new_node)
            node_mapping[a_node] = new_node
        new_node_set.output_layer = output_layer

        # Create new EdgeSet
        new_edge_set = EdgeSet()

        # Generate mutant edges
        mutant_edges = []
        for a_edge, b_edge, c_edge in zip(a_unrolled_edges, b_unrolled_edges, c_unrolled_edges):
            difference = ((b_edge.weight - c_edge.weight) * self.scaling_factor) + a_edge.weight
            if self.demo:
                print(f"(({b_edge.weight} - {c_edge.weight}) * {self.scaling_factor}) + {a_edge.weight} = {difference}")
            new_start = node_mapping[a_edge.start]
            new_end = node_mapping[a_edge.end]
            mutant_edge = Edge(new_start, new_end, difference)
            mutant_edges.append(mutant_edge)
        new_edge_set.edges = mutant_edges

        mutant = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, None)
        mutant.node_set = new_node_set
        mutant.edge_set = new_edge_set

        self.mutant = mutant

    def crossover(self):
        """
        Method used to perform cross over between the mutant and target member.
        """
        if self.demo:
            print("\nCROSSOVER")

        child = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, self.current_id)
        self.current_id += 1

        # Edges
        for target_edge, mutant_edge, child_edge in zip(self.target_member.edge_set.edges, self.mutant.edge_set.edges, child.edge_set.edges):
            if random.random() <= self.binomial_crossover_probability:
                child_edge.weight = mutant_edge.weight
                if self.demo:
                    print("\tChose mutant value")
            else:
                child_edge.weight = target_edge.weight
                if self.demo:
                    print("\tChose target value")

        # Input layer
        for target_node, mutant_node, child_node in zip(self.target_member.node_set.input_layer, self.mutant.node_set.input_layer, child.node_set.input_layer):
            if random.random() <= self.binomial_crossover_probability:
                child_node.bias = mutant_node.bias
            else:
                child_node.bias = target_node.bias

        # Hidden layers
        for target_layer, mutant_layer, child_layer in zip(self.target_member.node_set.hidden_layers, self.mutant.node_set.hidden_layers, child.node_set.hidden_layers):
            for target_node, mutant_node, child_node in zip(target_layer, mutant_layer, child_layer):
                if random.random() <= self.binomial_crossover_probability:
                    child_node.bias = mutant_node.bias
                else:
                    child_node.bias = target_node.bias

        # Output layer
        for target_node, mutant_node, child_node in zip(self.target_member.node_set.output_layer, self.mutant.node_set.output_layer, child.node_set.output_layer):
            if random.random() <= self.binomial_crossover_probability:
                child_node.bias = mutant_node.bias
            else:
                child_node.bias = target_node.bias

        self.child = child

    def replacement(self):
        """
        Method used to replace the population.
        """

        # Calculate child and parent fitness
        child_fitness = self.child.forward_pass()
        parent_fitness = self.target_member.forward_pass()

        # Compare fitness
        if child_fitness > parent_fitness:
            self.population.remove(self.target_member)
            self.child.id = self.target_member.id
            self.population.append(self.child)
        else:
            pass

    def check_convergence(self):
        """
        Method used to check if the fitness of the population has converged.
        """
        self.generation += 1

        if self.generation < (self.population_size / 2):
            pass
        else:
            if self.generation >= 3000:
                self.converged = True

    def log_progress(self):
        """
        Method used to check progress of evolution in training
        """

        best_fitness = max(member.fitness for member in self.population)
        avg_fitness = sum(member.fitness for member in self.population) / len(self.population)
        # print(f"Generation {self.generation}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")
        return avg_fitness

    def train(self):
        """
        Method used to train the population.
        """
        fitness_values = []
        while not self.converged:
            for member in self.population:
                self.target_member = member
                self.selection()
                self.create_mutant()
                self.crossover()
                self.replacement()
                self.check_convergence()
                fitness_values.append(self.log_progress())

        return fitness_values

    def test(self):
        """
        Method used to test the most fit member of the population after training
        """

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
#         'scaling_factor': 0.5,
#         'binomial_crossover_probability': 0.5,
#         'num_hidden_nodes': 2,
#         'learning_rate': 0.01,
#     }

#     classification_ga = DifferentialEvolution(data=classification_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
#     classification_ga.train()
#     prediction, actual = classification_ga.test()

#     print('\n')

#     regression_ga = DifferentialEvolution(data=regression_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
#     regression_ga.train()
#     prediction, actual = regression_ga.test()

# main()
