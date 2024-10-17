import random
from node import *
from edge import *

class FeedForwardNetwork:
    def __init__(self, training_data, testing_data, num_hidden_layers, input_size, hidden_size, output_size, is_class=True):
        self.training_data = training_data
        self.testing_data = testing_data
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_class = is_class

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
