__author__ = "<Hayden Perusich>", "<Shane Costello>"

import random
import numpy as np
from node import NodeSet, Node
from edge import EdgeSet, Edge
from meta_data import *

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

    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters, _id):
        self.id = _id
        self.is_class = data.is_class
        self.training_data = MetaData(data.get_training_set(hold_out_fold))
        self.testing_data = MetaData(data.get_test_set(hold_out_fold))
        self.num_hidden_layers = number_hidden_layers
        self.hidden_size = hyperparameters['num_hidden_nodes']
        self.input_size = data.num_features
        self.output_size = data.num_classes if self.is_class else 1
        self.classes = data.classes
        self.learning_rate = hyperparameters['learning_rate']
        self.fitness = 0

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
                input_sum = sum(
                    edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(node)) + node.bias
                node.value = input_sum
                node.value = tanh(node.value)

    def forward_output_layer(self):
        for out_node in self.node_set.output_layer:
            input_sum = sum(
                edge.weight * edge.start.value for edge in self.edge_set.get_incoming_edges(out_node)) + out_node.bias
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
                total_delta = sum(
                    edge.end.gradient_value * edge.weight for edge in self.edge_set.get_outgoing_edges(node))
                node.gradient_value = total_delta * tanh_derivative(node.value)
                for edge in self.edge_set.get_incoming_edges(node):
                    edge.weight += self.learning_rate * node.gradient_value * edge.start.value
                node.bias += self.learning_rate * node.gradient_value

    def train(self):
        loss_values = []
        for _ in range(EPOCHS):
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
        predictions, actual = [], []
        for point in self.testing_data.feature_vectors:
            prediction = self.forward(point)
            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(prediction)  # Returns the index of the highest value
            predicted_class = self.classes[predicted_class_index]  # Get the class label corresponding to that index
            actual = self.testing_data.target_vector[len(predictions)]  # Actual class label
            predictions.append(predicted_class)  # Append the predicted class label
            actual.append(actual)  # Append the actual class label
        return predictions, actual
    

    # unrolls the value from each node to create a int[]
    def unroll_nodes(self):
        nodes_list = []
        nodes = self.node_set

        # I input '*' into the nodes list to dictate where different layers are distinguished  
        for input_nodes in nodes.input_layer:
            nodes_list.append(input_nodes.bias)
        nodes_list.append('*') 

        for hidden_layers in nodes.hidden_layers:
            for hidden_node in hidden_layers:
                nodes_list.append(hidden_node.bias)
            nodes_list.append('*')

        for output_nodes in nodes.output_layer:
            nodes_list.append(output_nodes.bias)

        return nodes_list
        
    def unroll_edges(self):

        edge_list = []
        nodes = self.node_set
        edges = self.edge_set

        # I input '*' into the nodes list to dictate where different layers are distinguished  
        for input_nodes in nodes.input_layer:
                edge_list.append([edge.weight for edge in edges.get_outgoing_edges(input_nodes)])
        edge_list.append('*') 

        for hidden_layers in nodes.hidden_layers:
            for hidden_node in hidden_layers:
                edge_list.append([edge.weight for edge in edges.get_outgoing_edges(hidden_node)])
            edge_list.append('*')


        return edge_list

    def roll_up(self, node_list):
        node_set = NodeSet()
        edge_set = EdgeSet()

        # Input nodes
        input_layer = []
        while node_list and node_list[0] != '*':
            input_layer.append(Node(node_list.pop(0)))  
        if input_layer:
            node_set.input_layer = input_layer  

        # Output nodes
        output_layer = []
        while node_list and node_list[-1] != '*':
            output_layer.insert(0, Node(node_list.pop()))  
        if output_layer:
            node_set.output_layer = output_layer  

        # Hidden layers
        hidden_layers = []
        while node_list:
            hidden_layer = []
            while node_list and node_list[0] != '*':
                hidden_layer.append(Node(node_list.pop(0)))  
            if hidden_layer:
                hidden_layers.append(hidden_layer)  
            if node_list:
                node_list.pop(0)  
        node_set.hidden_layers = hidden_layers

        self.node_set = node_set



from root_data import *

def main():
    data = RootData('Project 4\data\soybean-small.data', True)

    

    ffn = FeedForwardNetwork(data=data, hold_out_fold=1, 
                            number_hidden_layers=2, hyperparameters={'num_hidden_nodes': 5, 'learning_rate' : 0.01},
                            _id=1)


#     class FeedForwardNetwork(
#     data: Any,
#     hold_out_fold: Any,
#     number_hidden_layers: Any,
#     hyperparameters: Any,
#     _id: Any
# )

    nodes = ffn.unroll_nodes()
    print(nodes)
    print(ffn.roll_up(nodes))
    print(ffn.unroll_nodes())




main()