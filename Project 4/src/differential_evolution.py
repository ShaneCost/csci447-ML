from feedforward_neural_network import *
import random
from node import *

class DifferentialEvolution:
    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters):
        self.population_size = hyperparameters['population_size']
        self.scaling_factor = hyperparameters['scaling_factor']
        self.binomial_crossover_probability = hyperparameters['binomial_crossover_probability']

        self.data = data
        self.hold_out_fold = hold_out_fold
        self.number_hidden_layers = number_hidden_layers
        self.hyperparameters = hyperparameters

        self.population = []
        self.generation = 1
        self.current_id = 0
        self.converged = False

        self.target_member = None
        self.a = None
        self.b = None
        self.c = None
        self.mutant = None
        self.child = None

        self.initialize_population()

    def initialize_population(self):
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

    def get_member(self, _id):
        for member in self.population:
            if member.id == _id:
                return member

    def create_mutant(self):
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

        # Input layer
        input_layer = []
        for a_node, b_node, c_node in zip(a_unrolled_nodes.input_layer, b_unrolled_nodes.input_layer, c_unrolled_nodes.input_layer):
            difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
            new_node = Node(difference)
            input_layer.append(new_node)
            node_mapping[a_node] = new_node  # Map old to new node
        new_node_set.input_layer = input_layer

        # Hidden layers
        hidden_layers = []
        for a_layer, b_layer, c_layer in zip(a_unrolled_nodes.hidden_layers, b_unrolled_nodes.hidden_layers, c_unrolled_nodes.hidden_layers):
            layer = []
            for a_node, b_node, c_node in zip(a_layer, b_layer, c_layer):
                difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
                new_node = Node(difference)
                layer.append(new_node)
                node_mapping[a_node] = new_node  # Map old to new node
            hidden_layers.append(layer)
        new_node_set.hidden_layers = hidden_layers

        # Output layer
        output_layer = []
        for a_node, b_node, c_node in zip(a_unrolled_nodes.output_layer, b_unrolled_nodes.output_layer, c_unrolled_nodes.output_layer):
            difference = ((b_node.bias - c_node.bias) * self.scaling_factor) + a_node.bias
            new_node = Node(difference)
            output_layer.append(new_node)
            node_mapping[a_node] = new_node  # Map old to new node
        new_node_set.output_layer = output_layer

        # Create new EdgeSet
        new_edge_set = EdgeSet()

        # Generate mutant edges
        mutant_edges = []
        for a_edge, b_edge, c_edge in zip(a_unrolled_edges, b_unrolled_edges, c_unrolled_edges):
            difference = ((b_edge.weight - c_edge.weight) * self.scaling_factor) + a_edge.weight
            new_start = node_mapping[a_edge.start]  # Use new node for start
            new_end = node_mapping[a_edge.end]  # Use new node for end
            mutant_edge = Edge(new_start, new_end, difference)
            mutant_edges.append(mutant_edge)
        new_edge_set.edges = mutant_edges

        mutant = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, None)
        mutant.node_set = new_node_set
        mutant.edge_set = new_edge_set

        self.mutant = mutant

    def crossover(self):
        child = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, self.current_id)
        self.current_id += 1

        for target_edge, mutant_edge, child_edge in zip(self.target_member.edge_set.edges, self.mutant.edge_set.edges, child.edge_set.edges):
            if random.random() <= self.binomial_crossover_probability:
                child_edge.weight = mutant_edge.weight
            else:
                child_edge.weight = target_edge.weight

        for target_node, mutant_node, child_node in zip(self.target_member.node_set.input_layer, self.mutant.node_set.input_layer, child.node_set.input_layer):
            if random.random() <= self.binomial_crossover_probability:
                child_node.bias = mutant_node.bias
            else:
                child_node.bias = target_node.bias

        for target_layer, mutant_layer, child_layer in zip(self.target_member.node_set.hidden_layers, self.mutant.node_set.hidden_layers, child.node_set.hidden_layers):
            for target_node, mutant_node, child_node in zip(target_layer, mutant_layer, child_layer):
                if random.random() <= self.binomial_crossover_probability:
                    child_node.bias = mutant_node.bias
                else:
                    child_node.bias = target_node.bias

        for target_node, mutant_node, child_node in zip(self.target_member.node_set.output_layer, self.mutant.node_set.output_layer, child.node_set.output_layer):
            if random.random() <= self.binomial_crossover_probability:
                child_node.bias = mutant_node.bias
            else:
                child_node.bias = target_node.bias

        self.child = child

    def replacement(self):
        child_fitness = self.child.forward_pass()
        parent_fitness = self.target_member.forward_pass()

        if child_fitness > parent_fitness:
            self.population.remove(self.target_member)
            self.child.id = self.target_member.id
            self.population.append(self.child)
        else:
            pass

    def check_convergence(self):
        self.generation += 1

        if self.generation < (self.population_size / 2):
            pass
        else:
            if self.generation >= 2000:
                self.converged = True

    def log_progress(self):
        best_fitness = max(member.fitness for member in self.population)
        avg_fitness = sum(member.fitness for member in self.population) / len(self.population)
        print(f"Generation {self.generation}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

    def train(self):
        while not self.converged:
            for member in self.population:
                self.target_member = member
                self.selection()
                self.create_mutant()
                self.crossover()
                self.replacement()
                self.check_convergence()
                self.log_progress()

    def test(self):
        sorted_population = sorted(self.population, key=lambda member: member.fitness, reverse=True)
        most_fit = sorted_population[0]
        prediction, actual = most_fit.test()

        return prediction, actual

from root_data import *
def main():
    classification = "../data/soybean-small.data"
    regression = "../data/forestfires.data"

    classification_data = RootData(path=classification, is_class=True)
    regression_data = RootData(path=regression, is_class=False)

    hyperparameters = {
        'population_size': 10,
        'scaling_factor': 0.5,
        'binomial_crossover_probability': 0.5,
        'num_hidden_nodes': 2,
        'learning_rate': 0.01,
    }

    classification_ga = DifferentialEvolution(data=classification_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
    classification_ga.train()
    prediction, actual = classification_ga.test()

    print('\n')

    regression_ga = DifferentialEvolution(data=regression_data, hold_out_fold=10, number_hidden_layers=1, hyperparameters=hyperparameters)
    regression_ga.train()
    prediction, actual = regression_ga.test()

# main()
