__author__ = "<Hayden Perusich>"

from feedforward_neural_network import *
import random
import math

class Patrical:
    def __init__(self, network, _id):
        
        self._id = _id
        self.network = network
        self.velocity = 0

        self.previous_posistion = [] #array of weight
        self.posistion = network.unroll_edges() #array of Edge objects    

        self.p_best = [] # personal best posistion array
        self.fitness_p_best =  0

    # special handeling for posistion array

    def get_posistion_array(self, array):
        posistion_array = []
        for x in array:
            posistion_array.append(x.get_weight())
        return posistion_array
     
    def update_posistion(self, x):
        network = self.network
        for i, value in enumerate(x):
            self.posistion[i].update_weight(value)

        # update the network to reflect the new weights
        self.netowrk = network.roll_up(network.unroll_nodes(), self.posistion)
      
    
    def get_fitness(self):
        self.network.forward_pass()
        fitness = self.network.fitness
        return fitness
    
    def print(self):
        print("Posistion:", self.get_posistion_array(self.posistion)[:5])
        print("Pbest:", self.get_posistion_array(self.p_best)[:5])
        print("Velocity:", self.velocity)


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

        self.g_best_network = None
        self.g_best = [] # global best posistion array
        self.fitness_g_best = 0
        
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

  
        self.update_p_best() # get fitness values of the population
        self.find_update_g_best() # get the bestg fitness values from the population
        


    def update_velocity(self, example=False):
        for member in self.population:
            # Get the current position and velocity
            p = np.array(member.get_posistion_array(member.posistion))  # current position (unrolled weight matrix)
            v_old = member.velocity  # current velocity

            # Random values between 0 and 1
            r1 = random.random()
            r2 = random.random()

            # Personal best position (Pbest)
            Pbest = np.array(member.get_posistion_array(member.p_best))

            # Global best position (Gbest)
            Gbest = np.array(member.get_posistion_array(self.g_best))

            # Update velocity formula
            new_velocity = (self.inertia) * v_old + (self.cognitive_update_rate) * r1 * (Pbest - p) + (self.social_update_rate) * r2 * (Gbest - p)
            new_velocity = np.clip(new_velocity, 0, 1)
            
            # Update the particle's velocity
            member.velocity = new_velocity

            if(example and member._id == 0):
                print(f"New calculated Velocity for 1st member is:")
                print(f"{member.velocity [:5]}...{member.velocity [-5:]}")



    def update_posisiton(self, example=False):
        for member in self.population:
                # Get the current position and velocity
                x = np.array(member.get_posistion_array(member.posistion))# current position (unrolled weight matrix)
                y = np.array(member.previous_posistion)
                v = member.velocity  # current velocity

                if y.shape == (0,):
                    new_position = v * (x)
                else:
                    new_position = (y) + v*(x)

                member.previous_posistion = x
                member.update_posistion(new_position)

                if(example and member._id == 0):
                    print("New posistion for 1st member is:")
                    print(f"{new_position [:5]}...{new_position [-5:]}")



    def update_p_best(self, example=False):
        for member in self.population:
            fitness =  member.get_fitness()
            if member.fitness_p_best < fitness:
                member.fitness_p_best = fitness
                member.p_best = member.posistion
                if (example and member._id == 0):
                   print(f"Found new Pbest for first member. Fitness: {member.p_best}")
    
            
    def find_update_g_best(self, example=False):
        fitness_g_best = self.fitness_g_best
        g_best_posistion = self.g_best

        for member in self.population:
            if fitness_g_best < member.fitness_p_best:
                g_best_posistion = member.p_best
                fitness_g_best = member.fitness_p_best
                best_network = member.network

                self.g_best_network = best_network
                self.fitness_g_best = fitness_g_best
                self.g_best = g_best_posistion
                if(example):
                    print(f"Found new Gbest. Fitness: {self.fitness_g_best}")


    def train(self, example=False):
        convergance = False
        count = 0
        fitness_values = []
        while not convergance:
            self.update_velocity(example=example)
            self.update_posisiton(example=example)
            self.update_p_best(example=example)

            pre_g_best_fitness = self.fitness_g_best
            self.find_update_g_best(example=example)

            if (pre_g_best_fitness == self.fitness_g_best):
                count+=1
                fitness_values.append(self.fitness_g_best)
                if(count > 3000):
                    convergance = True
                    # print("Final Fitness"self.fitness_g_best)
            else:
                count = 0

        return fitness_values



    def test(self):
        prediction, actual = self.g_best_network.test()
        return prediction, actual


# from root_data import *
# def main():

#     data = RootData('Project 4\data\soybean-small.data', True)
#     hyperparameter = {'population_size': 20, 'inertia': 0.4, 'cognitive_update_rate': 1.4, 
#                       'social_update_rate': 0.8, 'num_hidden_nodes':6}

#     swarm = PSO(data=data, hold_out_fold=1, number_hidden_layers=1, hyperparameters=hyperparameter)

#     swarm.train()
#     prediction, actual = swarm.test()

#     print(prediction)
#     print(actual)

# main()
