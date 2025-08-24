import numpy as np

from toynet.optimizers.protocol import Optimizer


class BasicOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, params: list[np.ndarray], grads: list) -> None:
        params[0][...] -= self.learning_rate * grads[0]
        params[1][...] -= self.learning_rate * grads[1]
