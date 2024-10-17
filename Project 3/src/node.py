class Node:
    def __init__(self, bias):
        self.value = 0
        self.bias = bias
        self.gradient_delta_value = 0

    def hyperbolic_tangent(self):
        pass

    def update_bias(self, new_bias):
        self.bias = new_bias

class NodeSet:
    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
