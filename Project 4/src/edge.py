__author__ = "<Shane Costello>"

class Edge:
    def __init__(self, start, end, weight):
        """
        Class used to represent an edge between two nodes

        :param start: Node on the left side of the edge
        :param end: Node on the right side of the edge
        :param weight: Initialized weight value
        """
        self.start = start
        self.end = end
        self.weight = weight

    def update_weight(self, new_weight):
        """
        Modify the weight of the edge

        :param new_weight: Integer representing the new weight
        """
        self.weight = new_weight
    
    def get_weight(self):
        """
        Method used to get weight value of edge

        :return: weight of edge
        """
        return self.weight

    def get_start(self):
        """
        Method used to get starting Node of the edge

        :return: Instance of node class representing the starting node
        """
        return self.start
    
    def get_end(self):
        """
        Method used to get ending Node of the edge

        :return: Instance of node class representing the ending node
        """
        return self.end
    
    def print_edge(self):
        """
        Method used to print an edge for debugging purposes
        """
        print("Start ", self.start.return_print_string() ," \nEnd ", self.end.return_print_string(), " \nWeight: ", self.weight, "\n")

    def print_weight(self):
        """
        Method used to print the weight for debugging purposes
        """
        print("Weight: ", self.weight)


class EdgeSet:
    def __init__(self):
        """
        Class used to represent a set of edges contained in a graph
        """
        self.edges = []

    def add(self, edge):
        """
        Method used to add an edge to a set of edges

        :param edge: Instance of the Edge class
        """
        self.edges.append(edge)
    
    def import_edges(self, edge_array):
        """
        Method used to convert a array into a list of edges

        :param edges: array of Edge Objects
        """
        self.edges = edge_array


    def get_incoming_edges(self, node):
        """
        Given a node, returns all edges feeding into it

        :param node: Instance of the Node class

        :return: All edges where the given node is the 'end' node
        """
        incoming_edges = []
        for edge in self.edges:
            if edge.end == node:
                incoming_edges.append(edge)

        return incoming_edges

    def get_outgoing_edges(self, node):
        """
        Given a node, returns all edges feeding out of it

        :param node: Instance of the Node class

        :return: All edges where the given node is the 'start' node
        """
        outgoing_edges = []
        for edge in self.edges:
            if edge.start == node:
                outgoing_edges.append(edge)

        return outgoing_edges

    def print_weights(self):
        """
        Method used to print all weights of the edge set for debugging purposes
        """
        string = ""
        for edge in self.edges:
            string += " "
            string += str(edge.get_weight())
            string += " "
        print(string)