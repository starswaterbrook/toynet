import numpy as np

from toynet.config import NUMERICAL_EPS
from toynet.functions.abstract import ActivationFunction


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        self._fn = lambda x: np.maximum(0, x)
        self._fn_derivative = lambda x: np.where(x > 0, 1, 0)


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        self._fn = lambda x: np.clip(1 / (1 + np.exp(-x)), NUMERICAL_EPS, 1 - NUMERICAL_EPS)
        self._fn_derivative = lambda x: self._fn(x) * (1 - self._fn(x))


class Identity(ActivationFunction):
    def __init__(self) -> None:
        self._fn = lambda x: x
        self._fn_derivative = lambda x: np.ones_like(x)


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        self._fn = self._num_safe_softmax
        self._fn_derivative = self._num_safe_softmax_deriv

    def _num_safe_softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)  # type: ignore[no-any-return]

    def _num_safe_softmax_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self._num_safe_softmax(x)
        return np.ones_like(s)
