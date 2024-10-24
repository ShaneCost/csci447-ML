class Edge:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

    def update_weight(self, new_weight):
        self.weight = new_weight
    
    def get_weight(self):
        return self.weight

    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    
    def print_edge(self):
        print("Edge: start",self.start," end: ", self.end," weight: ", self.weight)


class EdgeSet:
    def __init__(self):
        self.edges = []

    def add(self, edge):
        self.edges.append(edge)

    def get_incoming_edges(self, node):
        incoming_edges = []
        for edge in self.edges:
            if edge.end == node:
                incoming_edges.append(edge)

        return incoming_edges

    def get_outgoing_edges(self, node):
        outgoing_edges = []
        for edge in self.edges:
            if edge.start == node:
                outgoing_edges.append(edge)

        return outgoing_edges