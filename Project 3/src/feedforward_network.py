import random
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

    # funcrions to be used to train and test
    def back_prop(self):
        pass

    def forward(self, point):
        # Step 1: Set input layer values
        for index, feature in enumerate(point[:-1]):  # Assuming the last element is the label
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
                            input_sum += prev_node.activation() * edge.get_weight()  # Use activation of the previous node

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

        # Print the output layer activations
        for out_node in self.node_set.output_layer:
            out_node.print_output()
            out_node.print_node()


    # training and testing the FFN
    def train(self):
        pass

    def test(self):
        pass

from root_data import *

def main():
    data = RootData("Project 3\data\soybean-small.data", 'class')

    training = data.get_training_set(1)
    test = data.get_test_set(1)

    ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, data.num_classes)

    ffn.forward(training[0])
    print(training[-1])


main()