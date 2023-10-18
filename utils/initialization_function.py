import numpy as np


class InitializationFunction:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @staticmethod
    def xavier_initialization(self):
        limit = (6 / (self.num_inputs + self.num_outputs)) ** 0.5
        weights = []
        for i in range(self.num_outputs):
            weights = np.random.uniform(-limit, limit, size=(self.num_inputs, self.num_outputs))
        return weights
