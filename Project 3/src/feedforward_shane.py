import random
import numpy as np
from node import *
from edge import *

EPOCHS = 1000

def derivative_function (value):
    return 1 - np.tanh(value) ** 2

class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, hidden_size, input_size, output_size, classes, learning_rate,
                 is_class=True):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.is_class = is_class
        self.input_size = input_size
        if is_class:
            self.output_size = output_size
        else:
            self.output_size = 1
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
            return ((actual - self.node_set.regression_output) ** 2) / 2

    def calc_output_error(self, prediction, actual):
        if self.is_class:
            for node in self.node_set.output_layer:
                target = 1 if node.class_name == actual else 0
                error = ((target - node.value) ** 2) / 2
                delta = error * derivative_function(node.value)
                node.gradient_value = delta
        else:
            error = ((actual - prediction) ** 2) / 2
            delta = error * derivative_function(prediction)
            self.node_set.output_layer[0].gradient_value = delta

   # TODO: Finish this function to walk back through the graph, updating weights and biases
    # def walk_back(self):
    #     for layer in reversed(self.node_set.hidden_layers): # walk through layers backward
    #         for node in layer: # look at each node
    #             outgoing_edges = self.edge_set.get_outgoing_edges(node) # get all edges leaving that node
    #             for edge in outgoing_edges:
    #                 output_node_delta = edge.end.gradient_value # get the delta value of all connected output nodes
    #             # TODO: Find how to update weight + bias at each node, using the output nodes delta value calculated in calc_output_error()
    #                 edge.weight = self.learning_rate * node.gradient_value * edge.start.value

    #     # TODO: Handle input layers, consider case of 0 hidden layers
    #     for layer in self.node_set.input_layer:
    #         for node in layer:
    #             pass

    def walk_back(self):
        # Handle output layer first
        for node in self.node_set.output_layer:
            outgoing_edges = self.edge_set.get_outgoing_edges(node)
            for edge in outgoing_edges:
                # Update the weight
                edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                
                # Update the bias
                node.bias += self.learning_rate * node.gradient_value

        # Handle hidden layers
        for layer in reversed(self.node_set.hidden_layers):
            for node in layer:
                outgoing_edges = self.edge_set.get_outgoing_edges(node)
                total_delta = 0
                for edge in outgoing_edges:
                    # Make sure edge.end refers to the correct node (the one receiving the gradient)
                    total_delta += edge.end.gradient_value * edge.weight
                
                # Calculate gradient for current node
                node.gradient_value = total_delta * derivative_function(node.value)

                # Update weights and bias for incoming edges
                incoming_edges = self.edge_set.get_incoming_edges(node)
                for edge in incoming_edges:
                    edge.weight += self.learning_rate * node.gradient_value * edge.start.value

                # Update the bias for the current node
                node.bias += self.learning_rate * node.gradient_value

        # Handle input layer only if no hidden layers
        if self.num_hidden_layers == 0:
            for node in self.node_set.input_layer:
                # Typically, input nodes do not have weights to update
                pass  # No updates needed for input nodes

    def train(self):
        for epoch in range(EPOCHS):
            for cur, point in enumerate(self.training_data.feature_vectors):
                prediction = self.forward(point)  # push a point forward through the graph
                actual = self.training_data.target_vector[cur]  # get actual value
                loss_function_value = self.loss(actual)  # derive value of loss function

                # Back Propagation
                self.calc_output_error(prediction, actual)  # calculate the error at the output layer
                self.walk_back()  # update weights and biases

                # If you want to track loss, you could accumulate it here
                # total_loss += loss_function_value

            # Optionally log average loss per epoch here
            # print(f"Epoch {epoch}: Average Loss = {total_loss / len(self.training_data.feature_vectors):.4f}")



    def test(self):
        i = 0
        prediction = []
        actual = []
        for point in self.testing_data.feature_vectors:
            predict = self.forward(point)
            act = self.testing_data.target_vector[i]
            prediction.append(predict)
            actual.append(act)
            i += 1
        return prediction, actual

from root_data import *
from meta_data import *

def main():
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = RootData("Project 3\data\soybean-small.data")
        total_correct = 0
        total_predictions = 0
        avg = 0
        min_v = 100
        max_v = 0
        for fold in folds:

            training = MetaData(data.get_training_set(fold))
            test = MetaData(data.get_test_set(fold))

            ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, data.num_classes, data.classes, 0.01)
            ffn.train()

            prediction, actual = ffn.test()

            correct_predictions = sum(1 for pred, act in zip(prediction, actual) if pred == act)
            total_correct += correct_predictions
            total_predictions += len(actual)

             # Print results for each fold
            print(f"Fold {fold}: Accuracy = {correct_predictions}/{len(actual)} ({(correct_predictions / len(actual)) * 100:.2f}%)")

         # Calculate and print total accuracy across all folds
        total_accuracy = total_correct / total_predictions * 100 if total_predictions > 0 else 0
        print(f"Total Accuracy across all folds: {total_accuracy:.2f}%")
                



main()