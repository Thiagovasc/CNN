from typing import List


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

