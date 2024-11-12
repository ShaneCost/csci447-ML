__author__ = "<Shane Costello>"

from root_data import *
from meta_data import *
from feedforward_shane import *
from loss import *

hyperparameter_values = {
    'soybean-small': {
            0: {
                'num_nodes': 31,
                'learning_rate': 0.0244905510204
            },
            1: {
                'num_nodes': 25,
                'learning_rate': .01976302731
            },
            2: {
                'num_nodes': 23,
                'learning_rate': .0364031483
            }
    },
    'forestfires' : {
            0: {
                'num_nodes': 26,
                'learning_rate': 0.065306469
            },
            1: {
                'num_nodes': 11,
                'learning_rate': 0.041283924
            },
            2: {
                'num_nodes': 10,
                'learning_rate': 0.0125829273
            },
    }
}

def requirement_1(data_sets, holdout_fold, graph_sizes):
    """
    Provide sample outputs from one test fold showing performance on one classification network and one
    regression network. Show results for each of the cases where you have no hidden layers, one hidden
    layer, and two hidden layers.
    """

    for data in data_sets:
        print("CLASSIFICATION\n") if data.is_class else print("\nREGRESSION\n")

        train = MetaData(data.get_training_set(holdout_fold))
        test = MetaData(data.get_test_set(holdout_fold))

        for graph_size in graph_sizes:
            print("\t\nNUMBER OF HIDDEN LAYERS: ", graph_size)
            num_hidden_nodes = hyperparameter_values[data.name][graph_size]['num_nodes']
            learning_rate = hyperparameter_values[data.name][graph_size]['learning_rate']

            ffn = FeedForwardNetwork(train, test, graph_size, num_hidden_nodes, data.num_features, data.num_classes,
                                     data.classes, learning_rate, data.is_class)
            print('\t\ttraining...')
            ffn.train()
            print('\t\ttesting...\n')
            prediction, actual = ffn.test()

            loss = Loss(actual, prediction, data.is_class)

            if data.is_class:
                loss.confusion_matrix.print_confusion_matrix()
            else:
                print('Mean Squared Error')
                print(loss.results)



def requirement_2(data_sets, holdout_fold, graph_sizes):
    """
    Show a sample model for the smallest of each of your neural network types. This will consist of showing
    the weight matrices with the inputs/outputs of the layer labeled in some way.
    """

    graph_size = 0

    for data in data_sets:
        print("\nCLASSIFICATION\n") if data.is_class else print("\nREGRESSION\n")

        train = MetaData(data.get_training_set(holdout_fold))
        test = MetaData(data.get_test_set(holdout_fold))

        num_hidden_nodes = hyperparameter_values[data.name][graph_size]['num_nodes']
        learning_rate = hyperparameter_values[data.name][graph_size]['learning_rate']

        ffn = FeedForwardNetwork(train, test, graph_size, num_hidden_nodes, data.num_features, data.num_classes,
                                 data.classes, learning_rate, data.is_class)
        print('\ttraining...')
        ffn.train()

        print("\nSAMPLE TRAINED MODEL\n")
        edges = ffn.edge_set.edges
        for edge in edges:
            edge.print_edge()


def requirement_3(data_sets, holdout_fold, graph_sizes):
    """
    Demonstrate and explain how an example is propagated through a two hidden layer network of your
    choice. Be sure to show the activations at each layer being calculated correctly.
    """

    graph_size = 2

    classification_data = data_sets[0]

    train = MetaData(classification_data.get_training_set(holdout_fold))
    test = MetaData(classification_data.get_test_set(holdout_fold))

    num_hidden_nodes = 5
    learning_rate = hyperparameter_values[classification_data.name][graph_size]['learning_rate']

    ffn = FeedForwardNetwork(train, test, graph_size, num_hidden_nodes, classification_data.num_features,
                             classification_data.num_classes, classification_data.classes, learning_rate,
                             classification_data.is_class)

    point = train.feature_vectors[0]

    print("\nINPUT LAYER\n")
    # Step 1: Set input layer values
    for index, feature in enumerate(point):  # Assuming the last element is the label
        ffn.node_set.input_layer[index].update_value(feature)
        print('\tNode Value: ', feature)

    count = 1
    other = 1
    # Step 2: Iterate over hidden layers
    for layer in ffn.node_set.hidden_layers:
        print(f"\nHIDDEN LAYER {count}\n")
        for current_node in layer:
            print(f"\n\tNode {other}")
            # Calculate the weighted sum of inputs from the previous layer
            input_sum = 0
            incoming_edges = ffn.edge_set.get_incoming_edges(current_node)
            print(f"\n\t\tInputs...\n")
            print(f"\n\t\tnode value * weight")
            for edge in incoming_edges:
                inp_value = (edge.weight * edge.start.value)
                print(f"\t\t{edge.weight} * {edge.start.value}")
                input_sum += inp_value
            print(f"\t\tBias = {current_node.bias}")
            input_sum += current_node.bias
            current_node.value = input_sum
            current_node.activation()
            print(f"\n\t\tActivated Value = tanh({input_sum})")
            print(f"\t\t\t{current_node.value}")
            other += 1
        count += 1

    # Step 3: Forward to output layer
    print("\nOUTPUT LAYER\n")
    for out_node in ffn.node_set.output_layer:
        print(f'\n\t Assigned Class: {out_node.class_name}')
        input_sum = 0
        incoming_edges = ffn.edge_set.get_incoming_edges(out_node)
        print(f"\n\t\tInputs...")
        for edge in incoming_edges:
            inp_value = (edge.weight * edge.start.value)
            print(f"\t\t{edge.weight} * {edge.start.value}")
            input_sum += inp_value
        print(f"\t\tBias = {out_node.bias}")
        input_sum += out_node.bias
        out_node.value = input_sum
        print(f'\n\t\tWeighted Input Sum = {input_sum}')

    if ffn.is_class:
        print('\n\tSoftmax Activation\n')
        ffn.node_set.print_and_run_soft_max()
        prediction = max(ffn.node_set.soft_max_values, key=ffn.node_set.soft_max_values.get)
        print("\nPrediction: " + str(prediction))


def requirement_4(data_sets, holdout_fold, graph_sizes):
    """
    Demonstrate the gradient calculation at the output for one classification network and one regression
    network.
    """

    graph_size = 1

    for data in data_sets:
        print("\nCLASSIFICATION\n") if data.is_class else print("\nREGRESSION\n")

        train = MetaData(data.get_training_set(holdout_fold))
        test = MetaData(data.get_test_set(holdout_fold))

        num_hidden_nodes = hyperparameter_values[data.name][graph_size]['num_nodes']
        learning_rate = hyperparameter_values[data.name][graph_size]['learning_rate']

        ffn = FeedForwardNetwork(train, test, graph_size, num_hidden_nodes, data.num_features, data.num_classes,
                                 data.classes, learning_rate, data.is_class)

        point = train.feature_vectors[0]
        prediction = ffn.forward(point)
        actual = train.target_vector[0]
        actual = float(actual) if not data.is_class else actual

        print(f'\tPrediction: {prediction}\n\tActual: {actual}\n')

        if ffn.is_class:
            for node in ffn.node_set.output_layer:
                target = 1 if node.class_name == actual else 0
                error = ((target - node.value) ** 2) / 2
                print(f'\t\t{node.class_name} error = (({target} - {node.value})^2) / 2')
                print(f'\t\t{node.class_name} delta = {error} * sech({node.value})^2\n')
        else:
            error = ((actual - prediction) ** 2) / 2
            print(f'\t\terror = (({actual} - {prediction})^2) / 2')
            print(f'\t\tdelta = {error} * sech({prediction})^2\n')


def requirement_5(data_sets, holdout_fold, graph_sizes):
    """
    Demonstrate the weight updates occurring on a two-layer network for each of the layers
    for one classification network and one regression network.
    """

    graph_size = 2

    for data in data_sets:
        print("\nCLASSIFICATION\n") if data.is_class else print("\nREGRESSION\n")

        train = MetaData(data.get_training_set(holdout_fold))
        test = MetaData(data.get_test_set(holdout_fold))

        num_hidden_nodes = 5
        learning_rate = hyperparameter_values[data.name][graph_size]['learning_rate']

        ffn = FeedForwardNetwork(train, test, graph_size, num_hidden_nodes, data.num_features, data.num_classes,
                                 data.classes, learning_rate, data.is_class)

        point = train.feature_vectors[0]
        prediction = ffn.forward(point)
        actual = train.target_vector[0]
        actual = float(actual) if not data.is_class else actual
        ffn.calc_output_error(prediction, actual)

        # Handle output layer first
        print("\tOUTPUT LAYER\n")
        print("\tweight += learning rate * gradient * connecting nodes activation value")
        for node in ffn.node_set.output_layer:
            outgoing_edges = ffn.edge_set.get_outgoing_edges(node)
            for edge in outgoing_edges:
                # Update the weight
                edge.weight += ffn.learning_rate * node.gradient_value * edge.start.value
                print(f"\t{edge.weight} += {ffn.learning_rate} * {node.gradient_value} * {edge.start.value}")

        # Handle hidden layers
        count = 1
        other = 10
        for layer in reversed(ffn.node_set.hidden_layers):
            print(f"\n\tHIDDEN LAYER {count}\n")
            for node in layer:
                print(f"\n\t\tNode {other}\n")
                outgoing_edges = ffn.edge_set.get_outgoing_edges(node)
                total_delta = 0
                print("\t\tTotal error = connecting node gradient value * weight")
                for edge in outgoing_edges:
                    # Make sure edge.end refers to the correct node (the one receiving the gradient)
                    total_delta += edge.end.gradient_value * edge.weight
                    print(f"\t\t{total_delta} += {edge.end.gradient_value} * {edge.weight}")

                # Calculate gradient for current node
                node.gradient_value = total_delta * derivative_function(node.value)
                print(f'\n\t\tnode gradient value = {total_delta} * sech({node.value})^2')

                # Update weights and bias for incoming edges
                incoming_edges = ffn.edge_set.get_incoming_edges(node)
                print("\n\t\tweight += learning rate * gradient * connecting nodes activation value")
                for edge in incoming_edges:
                    edge.weight += ffn.learning_rate * node.gradient_value * edge.start.value
                    print(f"\t\t{edge.weight} += {ffn.learning_rate} * {node.gradient_value} * {edge.start.value}")

                other -= 1

                # Update the bias for the current node
            count += 1


def requirement_6(data_sets, holdout_fold, graph_sizes):
    """
    Show the average performance over the ten folds for one of the data sets for each of the types of
    networks.
    """

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    classification_data = data_sets[0]

    for num_hidden_layers in graph_sizes:

        print(f"\n{num_hidden_layers} Hidden Layers\n")

        num_hidden_nodes = hyperparameter_values[classification_data.name][num_hidden_layers]['num_nodes']
        learning_rate = hyperparameter_values[classification_data.name][num_hidden_layers]['learning_rate']

        total_predicted = []
        total_actual = []

        print("\t...performing 10-fold cross-validation...")
        for fold in folds:
            train = MetaData(classification_data.get_training_set(fold))
            test = MetaData(classification_data.get_test_set(fold))

            ffn = FeedForwardNetwork(train, test, num_hidden_layers, num_hidden_nodes, classification_data.num_features,
                                     classification_data.num_classes, classification_data.classes, learning_rate,
                                     classification_data.is_class)

            ffn.train()
            prediction, actual = ffn.test()
            total_predicted.extend(prediction)
            total_actual.extend(actual)

        loss = Loss(total_actual, total_predicted)

        print("\nRESULTS\n")

        loss.confusion_matrix.print_confusion_matrix()


def main():

    data_sets = [
        RootData("../data/soybean-small.data", True),
        RootData("../data/forestfires.data", False)
    ]

    holdout_fold = 10

    graph_sizes = [0, 1, 2]

    input('\nBegin Demo\n')
    """
    Provide sample outputs from one test fold showing performance on one classification network and one
    regression network. Show results for each of the cases where you have no hidden layers, one hidden
    layer, and two hidden layers.
    """
    requirement_1(data_sets, holdout_fold, graph_sizes)

    input('\nContinue\n')
    """
    Show a sample model for the smallest of each of your neural network types. This will consist of showing
    the weight matrices with the inputs/outputs of the layer labeled in some way.
    """
    requirement_2(data_sets, holdout_fold, graph_sizes)

    input('\nContinue\n')
    """
    Demonstrate and explain how an example is propagated through a two hidden layer network of your
    choice. Be sure to show the activations at each layer being calculated correctly.
    """
    requirement_3(data_sets, holdout_fold, graph_sizes)

    input('\nContinue\n')
    """
    Demonstrate the gradient calculation at the output for one classification network and one regression
    network.
    """
    requirement_4(data_sets, holdout_fold, graph_sizes)

    input('\nContinue\n')
    """
    Demonstrate the weight updates occurring on a two-layer network for each of the layers
    for one classification network and one regression network.
    """
    requirement_5(data_sets, holdout_fold, graph_sizes)

    input('\nContinue\n')
    """
    Show the average performance over the ten folds for one of the data sets for each of the types of
    networks.
    """
    requirement_6(data_sets, holdout_fold, graph_sizes)

if __name__ == '__main__':
    main()