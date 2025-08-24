import numpy as np

from toynet.functions.abstract import ActivationFunction
from toynet.initalizers.abstract import Initializer
from toynet.initalizers.he import He
from toynet.layers.protocol import Layer


class Dense(Layer):
    def __init__(
        self,
        in_values_count: int,
        out_values_count: int,
        activation_fn: type[ActivationFunction],
        initializer: type[Initializer] = He,
    ) -> None:
        self._weights, self._bias = initializer()(in_values_count, out_values_count)

        self.grad_w_accum = np.zeros_like(self._weights)
        self.grad_b_accum = np.zeros_like(self._bias)

        self.activation_fn = activation_fn()
        self.z: np.ndarray = np.zeros((1, out_values_count))
        self.h: np.ndarray = np.zeros((1, out_values_count))
        self.last_input: np.ndarray = np.zeros((in_values_count, 1))

    def forward(self, layer_input: np.ndarray, apply_activation: bool = True) -> np.ndarray:
        F = lambda x: self.activation_fn(x) if apply_activation else x  # noqa: E731
        self.last_input = layer_input
        self.z = layer_input @ self._weights + self._bias
        self.h = F(self.z)
        return self.h

    def backpass(
        self,
        grad_input: np.ndarray,
    ) -> np.ndarray:
        grad_z = grad_input * self.activation_fn(self.z, derivative=True)
        grad_w = self.last_input.T @ grad_z
        grad_output = grad_z @ self._weights.T

        self.grad_w_accum += grad_w
        self.grad_b_accum += grad_z

        return grad_output  # type: ignore[no-any-return]

    def reset_accumulated_gradient(self) -> None:
        self.grad_w_accum.fill(0)
        self.grad_b_accum.fill(0)

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self._weights, self._bias]

    @property
    def gradients(self) -> list[np.ndarray]:
        return [self.grad_w_accum, self.grad_b_accum]
