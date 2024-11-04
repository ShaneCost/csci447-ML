__author__ = "<Shane Costello>"

import math
import numpy as np

class Node:
    def __init__(self, bias):
        """
        Class used to model a node in a graph

        :param bias: Node's initialized bias value
        """
        self.value = 0
        self.bias = bias
        self.gradient_value = 0
        self.class_name = None

    def hyperbolic_tangent(self):
        """
        Calculate and return the hyperbolic tangent of the node's value.
        """
        return math.tanh(self.value)

    def update_bias(self, new_bias):
        """
        Method used to update the bias of the node's value.
        """
        self.bias = new_bias
    
    def update_value(self, new_value):
        """
        Method used to update the value of the node
        :param new_value: Integer representing the new value of the node
        """
        self.value = float(new_value)
    
    def print_node(self):
        """
        Method used to print a node for debugging purposes
        """
        print(f"Node: value={self.value}, bias={self.bias}, gradient_delta={self.gradient_value}")

    def return_print_string(self):
        """
        Method used to get a string representation of the node
        """
        return f"Node: value={self.value}, bias={self.bias}, gradient_delta={self.gradient_value}"
    
    def print_output(self):
        """
        Method used to print the output of the node
        """
        print(f"Node: value={self.value * self.bias}")

    def activation(self):
        """
        Compute the activation using the hyperbolic tangent function.
        """
        self.value = self.hyperbolic_tangent()
        return self.value


class NodeSet:
    def __init__(self):
        """
        Class used to model the set of all nodes in a feedforward network.
        """
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        self.soft_max_values = {}
        self.regression_output = 0

    def soft_max(self):
        """
        Method used to calculate the soft max value at the output layer
        """
        values = {}
        # Calculate the exponential of the output values
        exp_values = np.exp([node.value for node in self.output_layer])

        # Calculate the denominator (sum of exponential)
        denominator = np.sum(exp_values)

        # Calculate softmax values
        for i, node in enumerate(self.output_layer):
            output = exp_values[i] / denominator
            values[node.class_name] = output
            node.value = output

        self.soft_max_values = values

    def print_and_run_soft_max(self):
        """
        Method used to run softmax activation on the output layer
        and print various statements as it runs, for debugging purposes
        """
        values = {}
        # Calculate the exponential of the output values
        exp_values = np.exp([node.value for node in self.output_layer])

        # Calculate the denominator (sum of exponential)
        denominator = np.sum(exp_values)

        # Calculate softmax values
        for i, node in enumerate(self.output_layer):
            output = exp_values[i] / denominator
            print(f'\t{node.class_name}: {exp_values[i]} / {denominator} = {output}')
            values[node.class_name] = output
            node.value = output

        self.soft_max_values = values


    def linear_activation(self):
        """
        Method used to perform linear activation on the output layer
        """
        output = self.output_layer[0].value
        self.regression_output = output
        self.output_layer[0].value = output
