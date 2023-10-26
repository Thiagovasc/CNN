from math import tanh, exp


class ActivationFunction:
    def __init__(self, vector: list[float]):
        self.vector = vector

    @staticmethod
    def linear_function(self) -> list[float]:
        return self.vector

    @staticmethod
    def sigmoid_function(self) -> list[float]:
        for _ in range(len(self.vector)):
            self.vector[_] = 1 / (1 + exp(-self.vector[_]))

        return self.vector

    @staticmethod
    def sigmoid_derivation(self) -> float:
        return self.sigmoid_function(self.vector) * (1 - self.sigmoid_function(self.vector))

    @staticmethod
    def binary_step(self) -> list[float]:
        # output (0 or 1)
        for _ in range(len(self.vector)):
            if self.vector[_] >= 0:
                self.vector[_] = 1
            else:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def hiperbolic_tangent(self) -> float:
        # output range (-1 to 1)
        return (exp(self.vector) - exp(-self.vector)) / (exp(self.vector) + exp(-self.vector))

    @staticmethod
    def hiperbolic_derivative(self) -> list[float]:
        # return 1 - tanh(self.vector) ** 2
        return self.vector

    @staticmethod
    def relu(self) -> list[float]:
        # output (0, n)
        for _ in range(len(self.vector)):
            if self.vector[_] < 0:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def relu_derivative(self) -> list[float]:
        # output (0,1)
        for _ in range(len(self.vector)):
            if self.vector[_] <= 0:
                self.vector[_] = 0

        return self.vector

    @staticmethod
    def dot_product(vector_x: list[float], vector_y: list[float]) -> float:
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
