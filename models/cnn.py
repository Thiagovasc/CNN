from typing import List, Callable
from utils.train_utils import ActivationFunctions


class Neuron:
    def __init__(self, weights: List[float], input_value: List[float], learning_rate: float,
                 activation_function: Callable[[ActivationFunctions], float],
                 activation_function_derived: Callable[[ActivationFunctions], float],
                 activation_function_name: str) -> None:
        self.weights = weights
        self.input_value = input_value
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_function_derived = activation_function_derived
        self.activation_function_name = activation_function_name

    def compute_output(self) -> float:
        dot_product = ActivationFunctions.dot_product(self.weights, self.input_value)
        activation_functions = ActivationFunctions(n=dot_product)
        output = self.activation_function(activation_functions)
        return output


my_neuron = Neuron([5, 2, 0], [0, 10, 5], 2.3,
                   ActivationFunctions.hiperbolic_tangent,
                   ActivationFunctions.hiperbolic_derivative,
                   'hiperbolic')

my_neuron2 = Neuron([5, 2, 0], [0, 10, 5], 2.3,
                    ActivationFunctions.sigmoid_function,
                    ActivationFunctions.sigmoid_derivation,
                    'sigmoid')

print(my_neuron.compute_output())
print(my_neuron2.compute_output())
