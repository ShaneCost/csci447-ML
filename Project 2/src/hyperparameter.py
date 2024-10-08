__author__ = "<Shane Costello>"

class Hyperparameter:
    def __init__(self, name, starting_value, max_value, step_value):
        """
        Class used to model Hyperparameters

        :param name: string name of the hyperparameter
        :param starting_value: float value to start at in tuning
        :param max_value: float cap value in the t
        :param step_value: float value of how much to increase value on each iteration of tuning
        """
        self.name = name
        self.starting_value = starting_value
        self.value = starting_value
        self.max_value = max_value
        self.step_value = step_value
        self.prev_performance = 0
        self.is_tuned = False

    def step(self, new_performance):
        """
        Function that steps up the hyperparameter value

        :param new_performance: float value performance metric for algorithm performance at the current hyperparameter value
        :return: None
        """
        self.check_performance(new_performance)
        if not self.is_tuned:
            self.value += round(self.step_value, 3)
            self.value = round(self.value, 3)

    def check_performance(self, new_performance):
        """
        Function that compares performance metrics to find optimal hyperparameter value

        :param new_performance: float value performance metric for algorithm performance at the current hyperparameter value
        :return: None
        """
        if self.value == self.starting_value: # If it is the first iteration of tuning
            self.prev_performance = round(new_performance, 3)
        elif self.value >= self.max_value: # If the hyperparameter has reached or surpassed its cap
            self.is_tuned = True
            self.value = self.max_value
        else:
            if round(new_performance, 3) < round(self.prev_performance, 3): # If the new performance metric is lower (worse) than the previous performance
                self.value -= self.step_value # Decrement to reach the previous value
                self.value = round(self.value, 3)
                self.is_tuned = True # Set boolean attribute to true
            else: # If the new performance is better or equal to the previous
                self.prev_performance = round(new_performance, 3)

