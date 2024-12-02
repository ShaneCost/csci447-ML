from feedforward_neural_network import *
import random
import math

class Patrical:
    def __init__(self, network, _id):
        
        self._id = _id
        self.network = network
        self.velocity = 0

        self.previous_posistion = 0 #array of weight
        self.posistion = network.unroll_edges() #array of Edge objects
        

        self.p_best = [] # personal best 
        self.loss_p_best = max

    def get_posistion(self):
        posistion_array = []
        for x in self.posistion:
            posistion_array.append(x.get_weight())
    
    # updates
    def update_velocity(self, x):
        self.velocity = x
    
    def update_posistion(self, x):
        network = self.netowrk
        for i, value in enumerate(x):
            self.posistion[i].update_weight(value)

        # update the network to reflect the new weights
        self.netowrk = network.roll_up(network.unroll_nodes(), self.posistion)
    
    def update_p_best(self, p):
        self.p_best = p

    def update_g_best(self, g):
        self.g_best = g
    
    def get_id(self):
        return self._id    
    
    def get_loss(self):
        total_loss = []
        predictions, actuals = self.network.test()
        for prediction, actual in zip(predictions, actuals):
            loss = self.network.loss(prediction, actual)
            total_loss.append(loss)

        loss = np.mean(total_loss)
        return loss



class PSO:
    def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters):

        self.population_size = hyperparameters['population_size']
        self.inertia = hyperparameters['inertia']
        self.cognitive_update_rate = hyperparameters['cognitive_update_rate']
        self.social_update_rate = hyperparameters['social_update_rate']
        self.num_hidden_nodes = hyperparameters['num_hidden_nodes']

        self.data = data
        self.hold_out_fold = hold_out_fold
        self.number_hidden_layers = number_hidden_layers

        self.population = []

        self.g_best = [] # global best
        self.loss_g_best = max
        
        self.initialize_population()
    
    def initialize_population(self):
        # Create all members of the population
        _id = 0
        for _ in range(self.population_size):
            neural_network = FeedForwardNetwork(self.data, self.hold_out_fold, 
                                                self.number_hidden_layers, {'num_hidden_nodes':self.num_hidden_nodes, 'learning_rate': 0.1},
                                                  _id)
            
            partical = Patrical(neural_network, _id)
            _id += 1
            self.population.append(partical)
        self.current_id = _id + 1

        # get fitness value of the population
        self.update_p_best()
        

    def update_velocity(self):
        for member in self.population:
            # Get the current position and velocity
            X = member.posistion  # current position (unrolled weight matrix)
            v_old = member.velocity  # current velocity

            # Random values between 0 and 1
            r1 = random.random()
            r2 = random.random()

            # Personal best position (Pbest)
            Pbest = member.p_best

            # Global best position (Gbest)
            Gbest = self.get_global_best_position()

            # Update the velocity
            new_velocity = (self.inertia * v_old) + \
                           (self.cognitive_update_rate * r1 * (Pbest - X)) + \
                           (self.social_update_rate * r2 * (Gbest - X))

            # Update the particle's velocity
            member.update_velocity(new_velocity)

    def update_posisiton(self):
        for member in self.population:
                # Get the current position and velocity
                x = member.get_posistion()# current position (unrolled weight matrix)
                y = member.previous_posistion
                v = member.velocity  # current velocity

                new_position = np.array(y) + v(np.array(x))

                member.previous_posistion = x
                member.update_posistion(new_position)

    def update_p_best(self):
        for member in self.population:
            loss =  member.get_loss()
            if member.loss_p_best > loss:
                member.loss_p_best = loss
                member.p_best = member.posistion
            
    
    def find_update_g_best(self):
        loss_g_best = self.loss_g_best
        g_best_posistion = self.g_best

        for member in self.population:
            if loss_g_best > member.loss_p_best:
                g_best_posistion = member.p_best
                loss_g_best = member.loss_p_best

        self.loss_g_best = loss_g_best
        self.g_best = g_best_posistion


    def train(self):
        pass
        
    def test(self):
        pass


from root_data import *


# class PSO:
#     def __init__(self, data, hold_out_fold, number_hidden_layers, hyperparameters):

#         self.population_size = hyperparameters['population_size']
#         self.inertia = hyperparameters['inertia']
#         self.cognitive_update_rate = hyperparameters['cognitive_update_rate']
#         self.social_update_rate = hyperparameters["social_update_rate"]

def main():

    data = RootData('Project 4\data\soybean-small.data', True)
    hyperparameter = {'population_size': 5, 'inertia': 0.4, 'cognitive_update_rate': 1.4, 
                      'social_update_rate': 1.4, 'num_hidden_nodes':5}

    swarm = PSO(data=data, hold_out_fold=1, number_hidden_layers=1, hyperparameters=hyperparameter)



main()