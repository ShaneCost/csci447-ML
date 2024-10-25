import math
import numpy as np

class Node:
    def __init__(self, bias):
        self.value = 0
        self.bias = bias
        self.gradient_delta_value = 0
        self.class_name = None
        self.sof

    def hyperbolic_tangent(self):
        """Calculate and return the hyperbolic tangent of the node's value."""
        return math.tanh(self.value)

    def update_bias(self, new_bias):
        self.bias = new_bias
    
    def update_value(self, new_value):
        self.value = new_value
    
    def print_node(self):
        print(f"Node: value={self.value}, bias={self.bias}, gradient_delta={self.gradient_delta_value}, activation={self.activation_value}")
    
    def print_output(self):
        print(f"Node: value={self.value * self.bias}")

    def activation(self):
        """Compute the activation using the hyperbolic tangent function."""
        self.activation_value = self.hyperbolic_tangent()
        return self.activation_value


class NodeSet:
    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        self.soft_max_values = {}

    def soft_max(self):
        values = {}
        # Calculate the exponential of the output values
        exp_values = np.exp([node.value for node in self.output_layer])

        # Calculate the denominator (sum of exponential)
        denominator = np.sum(exp_values)

        # Calculate softmax values
        for i, node in enumerate(self.output_layer):
            values[node.class_name] = exp_values[i] / denominator

        self.soft_max_values = values
