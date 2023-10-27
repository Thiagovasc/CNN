from math import tanh, exp
from typing import List


class ActivationFunction:
    def __init__(self, vector: List[float]):
        self.vector = vector

    @staticmethod
    def linear_function(self) -> List[float]:
        return self.vector

    @staticmethod
    def sigmoid(self) -> List[float]:
        return [round(1 / (1 + exp(-self.vector[_])), 4) for _ in range(len(self.vector))]

    @staticmethod
    def sigmoid_derivative(self) -> List[float]:
        derivatives = []
        for val in self.vector:
            sigmoid_val = 1 / (1 + exp(-val))
            derivatives.append((sigmoid_val * (1 - sigmoid_val)).__round__(4))

        return derivatives

    @staticmethod
    def binary_step(self) -> List[float]:
        # output (0 or 1)
        for _ in range(len(self.vector)):
            if self.vector[_] >= 0:
                self.vector[_] = 1
            else:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def hiperbolic_tangent(self) -> List[float]:
        # output range (-1 to 1)
        return [round(tanh(self.vector[_]), 4) for _ in range(len(self.vector))]

    @staticmethod
    def hiperbolic_derivative(self) -> List[float]:
        return [round(1 - tanh(self.vector[_]) ** 2, 4) for _ in range(len(self.vector))]

    @staticmethod
    def relu(self) -> List[float]:
        # output (0, n)
        for _ in range(len(self.vector)):
            if self.vector[_] < 0:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def relu_derivative(self) -> List[float]:
        # output (0,1)
        for _ in range(len(self.vector)):
            if self.vector[_] <= 0:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def dot_product(vector_x: List[float], vector_y: List[float]) -> float:
        result = 0.0
        len_x = len(vector_x)
        len_y = len(vector_y)

        if len_x > len_y:
            vector_y.extend([0.0] * (len_x - len_y))
        elif len_y > len_x:
            vector_x.extend([0.0] * (len_y - len_x))

        for i in range(len(vector_x)):
            result += vector_x[i] * vector_y[i]

        return result
