from typing import Protocol

import numpy as np

from toynet.functions.abstract import ActivationFunction
from toynet.initalizers.abstract import Initializer


class Layer(Protocol):
    activation_fn: ActivationFunction
    z: np.ndarray
    h: np.ndarray
    last_input: np.ndarray

    def __init__(
        self,
        in_values_count: int,
        out_values_count: int,
        activation_fn: ActivationFunction,
        initializer: Initializer,
    ) -> None: ...

    def forward(self, layer_input: np.ndarray, apply_activation: bool = True) -> np.ndarray: ...

    def backpass(self, grad_input: np.ndarray) -> np.ndarray: ...

    def reset_accumulated_gradient(self) -> None: ...

    @property
    def parameters(self) -> list[np.ndarray]: ...

    @property
    def gradients(self) -> list[np.ndarray]: ...
