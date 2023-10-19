from typing import List
import numpy as np


class InitializationFunction:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @staticmethod
    def uniform_xavier_initialization(self) -> List[float]:
        limit = (6 / (self.num_inputs + self.num_outputs)) ** (1 / 2)
        weights = []
        for i in range(self.num_outputs):
            weight_matrix = np.random.uniform(low=-limit,
                                              high=limit,
                                              size=(self.num_inputs, self.num_outputs))
            weights.append(weight_matrix)

        return weights

    @staticmethod
    def normal_xavier_initialization(self) -> List[float]:
        limit = (2 / self.num_inputs + self.num_outputs) ** (1 / 2)
        weights: list = []
        for i in range(self.num_outputs):
            weight_matrix = np.random.normal(loc=0.0,
                                             scale=limit,
                                             size=(self.num_inputs, self.num_outputs))
            weights.append(weight_matrix)

        return weights

    @staticmethod
    def uniform_kaiming_initialization(self) -> List[float]:
        # Typically used when the activation function was ReLU or PReLU
        limit = (6 / self.num_inputs) ** (1 / 2)
        weights = []
        for i in range(self.num_outputs):
            weight_matrix = np.random.uniform(low=-limit,
                                              high=limit,
                                              size=(self.num_inputs, self.num_outputs))
            weights.append(weight_matrix)

        return weights

    @staticmethod
    def normal_kaiming_initialization(self):
        limit = (2 / self.num_inputs) ** (1 / 2)
        weights = []
        for i in range(self.num_outputs):
            weight_matrix = (np.random.normal(loc=0.0,
                                              scale=limit,
                                              size=(self.num_inputs, self.num_outputs)))
            weights.append(weight_matrix)

        return weights

