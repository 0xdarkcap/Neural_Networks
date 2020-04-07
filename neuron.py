import math

import numpy as np


# activation function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def activate(inputs, weights):
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w
    # perform activation
    return sigmoid(h)


if __name__ == "__main__":
    inputs = np.array([0.5, 0.3, 0.2])
    weights = np.array([0.4, 0.7, 0.2])
    output = activate(inputs, weights)
    print(output)
