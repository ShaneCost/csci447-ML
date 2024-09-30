class Hyperparameter:
    def __init__(self, name, starting_value, step_value):
        self.name = name
        self.value = starting_value
        self.step_value = step_value
        self.value_performance = {}
        self.is_tuned = False

    def step(self, performance):
        self.value_performance[self.value] = performance
        self.value += self.step_value