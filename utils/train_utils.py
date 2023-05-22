from math import tanh, exp
from typing import List


class ActivationFunctions:
    def __init__(self, n: float):
        self.n = n

    @staticmethod
    def linear_function(self) -> float:
        return self.n

    @staticmethod
    def sigmoid_function(self) -> float:
        return 1 / (1 + exp(-self.n))

    @staticmethod
    def sigmoid_derivation(self) -> float:
        return self.sigmoid_function(self.n) * (1 - self.sigmoid_function(self.n))

    @staticmethod
    def binary_step(self) -> int:
        # output (0 or 1)
        return 1 if self.n >= 0 else 0

    @staticmethod
    def hiperbolic_tangent(self) -> float:
        # output range (-1 to 1)
        return (exp(self.n) - exp(-self.n)) / (exp(self.n) + exp(-self.n))

    @staticmethod
    def hiperbolic_derivative(self) -> float:
        return 1 - tanh(self.n) ** 2

    @staticmethod
    def relu(self):
        # output (0, n)
        return self.n if self.n > 0 else 0

    @staticmethod
    def relu_derivative(self):
        # output (0,1)
        return 1 if self.n >= 0 else 0

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
