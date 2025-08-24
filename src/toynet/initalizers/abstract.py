from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    @abstractmethod
    def initialize_weights(self, in_amount: int, out_amount: int) -> np.ndarray: ...

    @abstractmethod
    def initialize_bias(self, out_amount: int) -> np.ndarray: ...

    def __call__(self, in_amount: int, out_amount: int) -> tuple[np.ndarray, np.ndarray]:
        weights = self.initialize_weights(in_amount, out_amount)
        bias = self.initialize_bias(out_amount)
        return weights, bias
