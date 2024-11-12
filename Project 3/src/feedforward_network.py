__author__ = "<Hayden Perusich>", "<Shane Costello>"



import random
import numpy as np
from node import NodeSet, Node
from edge import EdgeSet, Edge

EPOCHS = 100
BATCH_SIZE = 32  

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_values = np.exp(x - np.max(x))  # Stability improvement for softmax
    return exp_values / exp_values.sum()

def softmax_derivative(output):
    return output * (1 - output)

class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, hidden_size, input_size, output_size, 
                 classes, learning_rate, is_class=True):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.is_class = is_class
        self.input_size = input_size
        self.output_size = output_size if is_class else 1
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


    def forward(self, point):
        # Set input layer values
        for index, feature in enumerate(point):
            self.node_set.input_layer[index].update_value(feature)

        # Forward pass through hidden layers
        self.forward_hidden_layers()

        # Forward pass to output layer
        return self.forward_output_layer()

    def forward_hidden_layers(self):
        for layer in self.node_set.hidden_layers:
            for node in layer:
                input_sum = sum(edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(node)) + node.bias
                node.value = input_sum
                node.value = tanh(node.value)

    def forward_output_layer(self):
        for out_node in self.node_set.output_layer:
            input_sum = sum(edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(out_node)) + out_node.bias
            out_node.value = input_sum

        if self.is_class:
            # Apply softmax for classification
            output_values = np.array([node.value for node in self.node_set.output_layer])
            softmax_values = softmax(output_values)
            return softmax_values
        else:
            return self.node_set.output_layer[0].value

    def loss(self, actual, prediction):
        if self.is_class:
            # Find the index of the actual class
            actual_index = self.classes.index(actual)
            # Use the predicted probability for the actual class
            return -np.log(prediction[actual_index])
        else:
            # Mean squared error for regression
            return 0.5 * (actual - prediction) ** 2


    def calc_output_error(self, prediction, actual):
        if self.is_class:
            # For classification, calculate error using softmax derivative
            for i, node in enumerate(self.node_set.output_layer):
                target = 1 if node.class_name == actual else 0
                error = target - prediction[i]
                node.gradient_value = error * softmax_derivative(prediction[i])
        else:
            error = actual - prediction
            self.node_set.output_layer[0].gradient_value = error * tanh_derivative(prediction)

    def walk_back(self):
        # Update output layer weights and biases
        for node in self.node_set.output_layer:
            for edge in self.edge_set.get_outgoing_edges(node):
                edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                node.bias += self.learning_rate * node.gradient_value

        # Update hidden layers
        for layer in reversed(self.node_set.hidden_layers):
            for node in layer:
                total_delta = sum(edge.end.gradient_value * edge.weight for edge in self.edge_set.get_outgoing_edges(node))
                node.gradient_value = total_delta * tanh_derivative(node.value)
                for edge in self.edge_set.get_incoming_edges(node):
                    edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                node.bias += self.learning_rate * node.gradient_value

    def train(self):
        loss_values = []
        for epoch in range(EPOCHS):
            batch_loss = []
            for batch in range(0, len(self.training_data.feature_vectors), BATCH_SIZE):
                batch_data = self.training_data.feature_vectors[batch: batch + BATCH_SIZE]
                batch_targets = self.training_data.target_vector[batch: batch + BATCH_SIZE]
                
                # Mini-batch training
                for point, target in zip(batch_data, batch_targets):
                    prediction = self.forward(point)
                    actual = float(target) if not self.is_class else target
                    batch_loss.append(self.loss(actual, prediction))
                    self.calc_output_error(prediction, actual)
                    self.walk_back()

            # Calculate average loss for the epoch
            loss_values.append(np.mean(batch_loss))
            # print(f"Epoch {epoch + 1}/{EPOCHS}: Loss = {loss_values[-1]:.4f}")

        return loss_values

    def test(self):
        predictions, actuals = [], []
        for point in self.testing_data.feature_vectors:
            prediction = self.forward(point)
            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(prediction)  # Returns the index of the highest value
            predicted_class = self.classes[predicted_class_index]  # Get the class label corresponding to that index
            actual = self.testing_data.target_vector[len(predictions)]  # Actual class label
            predictions.append(predicted_class)  # Append the predicted class label
            actuals.append(actual)  # Append the actual class label
        return predictions, actuals