from typing import List, Callable
from utils.train_utils import ActivationFunctions


class Neuron:
    def __init__(self, weights: List[float], input_value: List[float], learning_rate: float,
                 activation_function: Callable[[ActivationFunctions], float],
                 activation_function_derived: Callable[[ActivationFunctions], float],
                 activation_function_name: str,
                 delta: float) -> None:
        self.delta = delta
        self.weights = weights
        self.input_value = input_value
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_function_derived = activation_function_derived
        self.activation_function_name = activation_function_name
        self.delta = 0.0

    def compute_output(self) -> float:
        self.input_value[-1] = 1.0
        dot_product = ActivationFunctions.dot_product(self.weights, self.input_value)
        activation_functions = ActivationFunctions(n=dot_product)
        output = self.activation_function(activation_functions)

        return output
