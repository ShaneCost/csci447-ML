import random
import numpy as np
from node import *
from edge import *

class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, hidden_size, input_size, output_size, is_class=True):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.is_class = is_class
        self.input_size = input_size
        self.output_size = output_size
        self.node_set = 0
        self.edge_set = 0

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
        node_set.output_layer = output_layer

        # Initialize edges
        if self.num_hidden_layers == 0:
            for in_node in node_set.input_layer:
                for out_node in node_set.output_layer:
                    edge = Edge(in_node, out_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

        elif self.num_hidden_layers == 1:
            for in_node in node_set.input_layer:
                for h_node in node_set.hidden_layers:
                    edge = Edge(in_node, h_node, random.uniform(-0.1, 0.1))
                    edge_set.edges.append(edge)

            for h_node in node_set.hidden_layers:
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

    # fucntions for back_prop
    def forward(self, point):
        # Step 1: Set input layer values
        for index, feature in enumerate(point):  # Assuming the last element is the label
            self.node_set.input_layer[index].update_value(feature)

        # Step 2: Forward through hidden layers
        previous_layer = self.node_set.input_layer
        for layer in self.node_set.hidden_layers:
            for current_node in layer:
                # Calculate the weighted sum of inputs from the previous layer
                input_sum = 0
                for prev_node in previous_layer:
                    edges = self.edge_set.get_outgoing_edges(prev_node)
                    for edge in edges:
                        if edge.get_end() == current_node:  # Check if the edge leads to the current node
                            input_sum += prev_node.activation() * edge.get_weight()   # Use activation of the previous node

                # Update the current node value and calculate its activation
                input_sum += current_node.bias  # Add bias
                current_node.update_value(input_sum)  # Update node value
                current_node.activation()  # Calculate activation

            # Move to the current layer for the next iteration
            previous_layer = layer

        # Step 3: Forward to output layer
        for out_node in self.node_set.output_layer:
            input_sum = 0
            for h_node in self.node_set.hidden_layers[-1]:  # Last hidden layer
                edges = self.edge_set.get_outgoing_edges(h_node)
                for edge in edges:
                    if edge.get_end() == out_node:  # Check if the edge leads to the output node
                        input_sum += h_node.activation() * edge.get_weight()  # Use activation of the last hidden node

            # Update the output node value and calculate its activation
            input_sum += out_node.bias  # Add bias
            out_node.update_value(input_sum)  # Update node value

            out_node.print_node()


    def soft_max(self):
        pass

    def loss(self, y_pred, y_true, type='cross_entropy'):

        y_pred = np.array(y_pred, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)


        # cross_entropy loss
        if type == 'cross_entropy':
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        # mean squared error loss
        else:
            loss = np.mean((y_true - y_pred) ** 2)
        
        return loss
        

    def gradient_descent(self, mini_batch, learning_rate=0.01):
        for point in mini_batch:
            self.forward(point)

        # Collect predictions for loss computation
        y_preds = np.array([node.activation() for node in self.node_set.output_layer])
        loss = self.loss(y_preds, np.array([point[-1] for point in mini_batch]))

        self.back_prop(mini_batch, learning_rate)

        return loss

    # funcrions to be used to train and test
    def back_prop(self, mini_batch, learning_rate):
        output_layer_gradients = []
        for i, point in enumerate(mini_batch):
            y_pred = [node.activation() for node in self.node_set.output_layer]
            if self.is_class:
                gradient = y_pred - point[-1]
            else:
                gradient = 2 * (y_pred - point[-1]) / len(mini_batch)

            output_layer_gradients.append(gradient)

            for out_node in self.node_set.output_layer:
                for h_node in self.node_set.hidden_layers[-1]:
                    edge = self.edge_set.get_edge(h_node, out_node)
                    if edge:
                        weight_update = learning_rate * gradient * h_node.activation()
                        edge.update_weight(edge.get_weight() - weight_update)

                out_node.bias -= learning_rate * gradient

        # Backpropagate through hidden layers
        for layer_index in reversed(range(len(self.node_set.hidden_layers))):
            layer = self.node_set.hidden_layers[layer_index]
            next_layer_gradients = output_layer_gradients if layer_index == len(self.node_set.hidden_layers) - 1 else []

            for h_node in layer:
                gradient = sum(next_layer_gradients)

                for prev_node in self.node_set.hidden_layers[layer_index - 1] if layer_index > 0 else self.node_set.input_layer:
                    edge = self.edge_set.get_edge(prev_node, h_node)
                    if edge:
                        weight_update = learning_rate * gradient * prev_node.activation()
                        edge.update_weight(edge.get_weight() - weight_update)

                h_node.bias -= learning_rate * gradient


    # training and testing the FFN
    def train(self, epochs=100, learning_rate=0.01, batch_size=32):
        num_samples = len(self.training_data)
        labels = np.array([point[-1] for point in self.training_data])

        for epoch in range(epochs):
            np.random.shuffle(self.training_data)  # Shuffle data each epoch
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mini_batch = self.training_data[start:end]
                # Update mini_batch with one-hot encoded labels
                loss = self.gradient_descent(mini_batch, learning_rate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')


    def test(self):
        pass

from root_data import *
from meta_data import *

def main():
    data = RootData("Project 3\data\soybean-small.data")

    training = MetaData(data.get_training_set(1))
    test = MetaData(data.get_test_set(1))

    ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, data.num_classes)

    # ffn.train(epochs=100, learning_rate=0.01, batch_size=32)
    ffn.forward(training.feature_vectors[0])
    # print(training[-1])


main()