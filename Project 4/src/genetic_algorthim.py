from feedforward_neural_network import *
import random
import math

class GeneticAlgorithm:
    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters):
        self.population_size = hyperparameters['population_size']
        self.crossover_rate = hyperparameters['crossover_rate']
        self.mutation_rate = hyperparameters['mutation_rate']

        self.data = data
        self.hold_out_fold = hold_out_fold
        self.number_hidden_layers = number_hidden_layers
        self.hyperparameters = hyperparameters

        self.population = []
        self.current_id = 0
        self.current_parent_1 = None
        self.current_parent_2 = None
        self.current_child = None
        self.converged = False
        self.generation = 1

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

    def fitness_proportionate_selection(self):
        # Calculate total fitness
        total_fitness = sum(member.fitness for member in self.population)

        # Normalize fitness values to create probabilities
        if total_fitness == 0:
            weights = [1 / len(self.population)] * len(self.population)  # Uniform probabilities
        else:
            weights = [member.fitness / total_fitness for member in self.population]

        # Use random.choices() to select two parents
        parents = random.choices(self.population, weights=weights, k=2)

        self.current_parent_1 = parents[0]
        self.current_parent_2 = parents[1]

    def arithmetic_crossover(self):
        child = FeedForwardNetwork(self.data, self.hold_out_fold, self.number_hidden_layers, self.hyperparameters, self.current_id)
        self.current_id += 1

        for parent_1_edge, parent_2_edge, child_edge in zip(self.current_parent_1.edge_set, self.current_parent_2.edge_set, child.edge_set):
            child_edge.weight = ((parent_1_edge.weight + parent_2_edge.weight) / 2)

        for parent_1_node, parent_2_node, child_node in zip(self.current_parent_1.node_set, self.current_parent_2.node_set, child.node_set):
            child_node.bias = ((parent_1_node.bais + parent_2_node.bias) / 2)

        self.current_child = child

    def order_based_mutation(self):
        number_of_edge_mutations = math.ceil(len(self.current_child.edge_set) * self.mutation_rate)
        max_v = len(self.current_child.edge_set) - 1
        for i in range(number_of_edge_mutations):
            edge_1 = self.current_child.edge_set[random.randint(0, max_v)]
            edge_2 = self.current_child.edge_set[random.randint(0, max_v)]

            edge_1_weight = edge_1.weight
            edge_1.weight = edge_2.weight
            edge_2.weight = edge_1_weight

        number_of_node_mutations = math.ceil(len(self.current_child.node_set) * self.mutation_rate)
        max_v = len(self.current_child.node_set) - 1
        for i in range(number_of_node_mutations):
            node_1 = self.current_child.node_set[random.randint(0, max_v)]
            node_2 = self.current_child.node_set[random.randint(0, max_v)]

            node_1_bias = node_1.bais
            node_1.bias = node_2.bais
            node_2.bias = node_1_bias

    def read_state_replacement(self):
        # Find the two least fit members
        sorted_population = sorted(self.population, key=lambda member: member.fitness)
        least_fit_1 = sorted_population[0]
        least_fit_2 = sorted_population[1]

        # Replace the two least fit members with the new child
        self.population.remove(least_fit_1)
        self.population.remove(least_fit_2)

        # Add the new child to the population
        self.population.append(self.current_child)
        self.current_child.forward_pass() # Derive child's fitness value

        self.current_parent_1 = None
        self.current_parent_2 = None
        self.current_child = None

    def check_convergence(self):
        self.generation += 1

        if self.generation < (self.population_size/2):
            pass
        else:
            if self.generation >= 100:
                self.converged = True

    def log_progress(self):
        best_fitness = max(member.fitness for member in self.population)
        avg_fitness = sum(member.fitness for member in self.population) / len(self.population)
        print(f"Generation {self.generation}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

    def train(self):
        while not self.converged:
            self.fitness_proportionate_selection()
            self.arithmetic_crossover()
            self.order_based_mutation()
            self.read_state_replacement()
            self.check_convergence()
            self.log_progress()

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