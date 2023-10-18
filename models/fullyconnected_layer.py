from typing import Callable, List
from models.neuron import Neuron
from utils.train_utils import ActivationFunctions


class FullyConnectedLayer:
    def __init__(self, num_neurons: int, num_inputs: int, learning_rate: float,
                 activation_function: Callable[[ActivationFunctions], float],
                 activation_function_derived: Callable[[ActivationFunctions], float]):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_function_derived = activation_function_derived

        """    
        self.neurons = []
        for _ in range(num_neurons):
            weights = [0.0] * (num_inputs + 1)
            input_value = [0.0] * (num_inputs + 1)
            delta = 0.0
            neuron = Neuron(weights, input_value, learning_rate, activation_function,
                            activation_function_derived, activation_function_name, delta)
            self.neurons.append(neuron)
            """

    def forward_propagation(self, input_values: List[float]) -> List[float]:
        outputs = []
        for neuron in self.neurons:
            neuron.input_value = input_values
            output = neuron.compute_output()
            outputs.append(output)
        return outputs

    def backward_propagation(self, output_deltas: List[float]):
        for i, neuron in enumerate(self.neurons):
            neuron.delta = output_deltas[i] * neuron.activation_function_derived(
                ActivationFunctions(n=neuron.compute_output()))
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= neuron.learning_rate * neuron.delta * neuron.input_value[j]
