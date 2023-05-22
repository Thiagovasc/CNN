from typing import List


class LossFunction:
    def __init__(self, output_reached: List[float], output_predicted: List[float]):
        self.output_reached = output_reached
        self.output_predicted = output_predicted

    @staticmethod
    def mean_absolute_error(self) -> float:
        error_list: List[float] = []
        for i in range(len(self.output_reached)):
            error_list.append(abs(self.output_reached[i] - self.output_predicted[i]))

        return sum(error_list) / len(error_list)

    @staticmethod
    def mean_square_error(self) -> float:
        error_list: List[float] = []
        error: float
        for i in range(len(self.output_reached)):
            error = (self.output_reached[i] - self.output_predicted[i]) ** 2
            error_list.append(error)

        return sum(error_list) / len(error_list)
