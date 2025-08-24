import numpy as np

from toynet.initalizers.abstract import Initializer


class He(Initializer):
    def initialize_weights(self, in_amount: int, out_amount: int) -> np.ndarray:
        he_std = (2.0 / in_amount) ** 0.5
        return np.random.normal(0.0, he_std, (in_amount, out_amount))

    def initialize_bias(self, out_amount: int) -> np.ndarray:
        return np.zeros((1, out_amount))
