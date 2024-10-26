import random
import numpy as np
from node import *
from edge import *


class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, hidden_size, input_size, output_size, classes, learning_rate,
                 is_class=True):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.is_class = is_class
        self.input_size = input_size
        self.output_size = output_size
        self.classes = classes
        self.learning_rate = learning_rate

        self.node_set = NodeSet()
        self.edge_set = EdgeSet()

        self.initialize_graph()

    def initialize_graph(self):
        node_set = NodeSet()
        edge_set = EdgeSet()

        # Initialize input layer nodes
        input_layer = [Node(random.uniform(-0.1, 0.1)) for _ in range(self.input_size)]
        node_set.input_layer = input_layer

        # Initialize hidden layers (2D array of nodes)
        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            layer = [Node(random.uniform(-0.1, 0.1)) for _ in range(self.hidden_size)]
            hidden_layers.append(layer)
        node_set.hidden_layers = hidden_layers

        # Initialize output layer nodes
        output_layer = [Node(random.uniform(-0.1, 0.1)) for _ in range(self.output_size)]
        if self.is_class:
            for i in range(self.output_size):
                output_layer[i].class_name = self.classes[i]
        node_set.output_layer = output_layer

        # Initialize edges
        if self.num_hidden_layers == 0:
            for in_node in node_set.input_layer:
                for out_node in node_set.output_layer:
                    edge = Edge(in_node, out_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

        elif self.num_hidden_layers == 1:
            for in_node in node_set.input_layer:
                for h_node in node_set.hidden_layers[0]:
                    edge = Edge(in_node, h_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

            for h_node in node_set.hidden_layers[0]:
                for out_node in node_set.output_layer:
                    edge = Edge(h_node, out_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

        elif self.num_hidden_layers == 2:
            for in_node in node_set.input_layer:
                for h_node_1 in node_set.hidden_layers[0]:
                    edge = Edge(in_node, h_node_1, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

            for h_node_1 in node_set.hidden_layers[0]:
                for h_node_2 in node_set.hidden_layers[1]:
                    edge = Edge(h_node_1, h_node_2, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

            for h_node in node_set.hidden_layers[1]:
                for out_node in node_set.output_layer:
                    edge = Edge(h_node, out_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

        self.node_set = node_set
        self.edge_set = edge_set

    # function for forward propagation
    def forward(self, point):
        # Step 1: Set input layer values
        for index, feature in enumerate(point):  # Assuming the last element is the label
            self.node_set.input_layer[index].update_value(feature)

        # Step 2: Iterate over hidden layers
        for layer in self.node_set.hidden_layers:
            for current_node in layer:
                # Calculate the weighted sum of inputs from the previous layer
                input_sum = 0
                incoming_edges = self.edge_set.get_incoming_edges(current_node)
                for edge in incoming_edges:
                    inp_value = (edge.weight * edge.start.value)
                    input_sum += inp_value
                input_sum += current_node.bias
                current_node.value = input_sum
                current_node.activation()

        # Step 3: Forward to output layer
        for out_node in self.node_set.output_layer:
            input_sum = 0
            incoming_edges = self.edge_set.get_incoming_edges(out_node)
            for edge in incoming_edges:
                inp_value = (edge.weight * edge.start.value)
                input_sum += inp_value
            input_sum += out_node.bias
            out_node.value = input_sum

        if self.is_class:
            self.node_set.soft_max()
            prediction = max(self.node_set.soft_max_values, key=self.node_set.soft_max_values.get)
        else:
            self.node_set.linear_activation()
            prediction = self.node_set.regression_output

        return prediction

    def loss(self, actual):
        # Cross-Entropy Loss
        if self.is_class:
            # Ensure probabilities are between 0 and 1
            predicted_probs = {k: np.clip(v, 1e-15, 1 - 1e-15) for k, v in self.node_set.soft_max_values.items()}
            # Cross-entropy formula for multiclass classification
            return -np.log(predicted_probs[actual])
        # Mean Squared Error
        else:
            return (actual - self.node_set.regression_output) ** 2

    def calc_error_values(self, prediction, actual):
        if self.is_class:
            for node in self.node_set.output_layer:
                class_name = node.class_name
                probability_value = self.node_set.soft_max_values[class_name]
                if class_name == actual:
                    gradient = probability_value - 1
                else:
                    gradient = probability_value
                node.gradient_delta_value = gradient
        else:
            gradient = prediction - actual
            self.node_set.output_layer[0].gradient_delta_value = gradient

        for layer in self.node_set.hidden_layers:
            for node in layer:
                total_error = 0
                out_going_nodes = self.edge_set.get_outgoing_edges(node)
                for edge in out_going_nodes:
                    weight = edge.weight
                    connecting_nodes_error = edge.end.gradient_delta_value
                    derivative_function_value = 1 - np.tanh(node.value) ** 2
                    error = weight * connecting_nodes_error * derivative_function_value
                    total_error += error
                node.gradient_delta_value = total_error

    def update_weights(self):
        for edge in self.edge_set.edges:
            start_activation = edge.start.activation()
            end_error = edge.end.gradient_delta_value
            weight = edge.weight
            learning_rate = self.learning_rate
            new_weight = weight - (learning_rate * start_activation * end_error)
            edge.update_weight(new_weight)


    def train(self):
        i = 0
        correct = 0
        for point in self.training_data.feature_vectors:
            prediction = self.forward(point) # push a point forward through the graph
            actual = self.training_data.target_vector[i] # get actual value
            loss_function_value = self.loss(actual) # derive value of loss function
            self.calc_error_values(prediction, actual) # calculate the error at the output layer
            self.update_weights()
            # print("actual:", actual)
            # print("predicted:", prediction)
            # print("loss:", loss_function_value)
            # print("\n")
            if actual == prediction:
                correct += 1
            i += 1
        return correct

from root_data import *
from meta_data import *

def main():
    for i in range(100):
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = RootData("../data/soybean-small.data")

        for fold in folds:

            training = MetaData(data.get_training_set(fold))
            test = MetaData(data.get_test_set(fold))

            ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, data.num_classes, data.classes, 0.01)

            print(ffn.train())

main()