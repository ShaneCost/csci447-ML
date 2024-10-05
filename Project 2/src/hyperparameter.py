class Hyperparameter:
    def __init__(self, name, starting_value, max_value, step_value):
        self.name = name
        self.starting_value = starting_value
        self.value = starting_value
        self.max_value = max_value
        self.step_value = step_value
        self.prev_performance = 0
        self.is_tuned = False

    def step(self, new_performance):
        self.check_performance(new_performance)
        if not self.is_tuned:
            self.value += round(self.step_value, 3)
            self.value = round(self.value, 3)

    def check_performance(self, new_performance):
        if self.value == self.starting_value:
            self.prev_performance = round(new_performance, 3)
        elif self.value >= self.max_value:
            self.is_tuned = True
            self.value = self.max_value
        else:
            if round(new_performance, 3) < round(self.prev_performance, 3):
                print(new_performance, " is less than ", self.prev_performance)
                self.value -= self.step_value
                self.value = round(self.value, 3)
                self.is_tuned = True
            else:
                self.prev_performance = round(new_performance, 3)

