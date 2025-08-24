from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None: ...

    def step(self, params: list[np.ndarray], grads: list) -> None: ...
