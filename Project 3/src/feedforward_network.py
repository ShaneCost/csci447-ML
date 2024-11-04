__author__ = "<Hayden Perusich>"

import random
import numpy as np
from node import *
from edge import *

EPOCHS = 100

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)  # Derivative of ReLU

class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, hidden_size, input_size, output_size,
                 learning_rate, is_class=False):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.is_class = is_class
        self.input_size = input_size
        self.output_size = output_size
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

        # Initialize hidden layers
        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            layer = [Node(random.uniform(-0.1, 0.1)) for _ in range(self.hidden_size)]
            hidden_layers.append(layer)
        node_set.hidden_layers = hidden_layers

        # Initialize output layer nodes
        output_layer = [Node(random.uniform(-0.1, 0.1)) for _ in range(self.output_size)]
        node_set.output_layer = output_layer

        # Initialize edges
        previous_layer = node_set.input_layer
        for layer in hidden_layers + [output_layer]:
            for in_node in previous_layer:
                for out_node in layer:
                    edge = Edge(in_node, out_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)
            previous_layer = layer

        self.node_set = node_set
        self.edge_set = edge_set

    def forward(self, point):
        # Set input layer values
        for index, feature in enumerate(point):
            self.node_set.input_layer[index].update_value(float(feature))  # Convert to float

        # Forward pass through hidden layers
        for layer in self.node_set.hidden_layers:
            for current_node in layer:
                input_sum = sum(edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(current_node))
                input_sum += current_node.bias
                current_node.value = relu(input_sum)  # Use ReLU activation

        # Forward to output layer
        output_values = []
        for out_node in self.node_set.output_layer:
            input_sum = sum(edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(out_node))
            input_sum += out_node.bias
            out_node.value = input_sum  # For regression, output should be raw value
            output_values.append(float(out_node.value))  # Ensure it's float

        return output_values


    def loss(self, actual):
        # Mean Squared Error
        predictions = [node.value for node in self.node_set.output_layer]
        return np.mean((np.array(predictions) - actual) ** 2)

    def calc_output_error(self, predictions, actual):
        for i, node in enumerate(self.node_set.output_layer):
            error = actual[i] - predictions[i]
            node.gradient_value = error * 1  # For MSE, derivative is 1

    def walk_back(self):
        # Update weights and biases for output layer
        for node in self.node_set.output_layer:
            outgoing_edges = self.edge_set.get_outgoing_edges(node)
            for edge in outgoing_edges:
                edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                node.bias += self.learning_rate * node.gradient_value

        # Update hidden layers
        for layer in reversed(self.node_set.hidden_layers):
            for node in layer:
                outgoing_edges = self.edge_set.get_outgoing_edges(node)
                total_delta = sum(edge.end.gradient_value * edge.weight for edge in outgoing_edges)
                node.gradient_value = total_delta * relu_derivative(node.value)

                incoming_edges = self.edge_set.get_incoming_edges(node)
                for edge in incoming_edges:
                    edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                    node.bias += self.learning_rate * node.gradient_value

    def train(self):
        for epoch in range(EPOCHS):
            total_loss = 0
            for cur, point in enumerate(self.training_data.feature_vectors):
                # Ensure actual values are in float format
                actual = np.array([float(self.training_data.target_vector[cur])])
                predictions = self.forward(point)
                loss_value = self.loss(actual)
                total_loss += loss_value

                self.calc_output_error(predictions, actual)
                self.walk_back()

            # print(f"Epoch {epoch}: Average Loss = {total_loss / len(self.training_data.feature_vectors):.4f}")


    def test(self):
        predictions = []
        for i, point in enumerate(self.testing_data.feature_vectors):
            predict = self.forward(point)
            predictions.extend(predict)

        actuals = np.array(self.testing_data.target_vector, dtype=np.float64)
        predictions = np.array(predictions, dtype=np.float64)
        return predictions, actuals



# from root_data import *
# from meta_data import *

# def main():
#     folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     data = RootData("Project 3/data/machine.data", False)

#     for fold in folds:
#         training = MetaData(data.get_training_set(fold))
#         test = MetaData(data.get_test_set(fold))

#         ffn = FeedForwardNetwork(training, test, num_hidden_layers=1, hidden_size=5,
#                                  input_size=data.num_features, output_size=1,
#                                  learning_rate=0.01, is_class=False)
#         print("Graph created")
#         ffn.train()


#         predictions, actuals = ffn.test()
#         print(actuals)
#         print(predictions)

# if __name__ == "__main__":
#     main()