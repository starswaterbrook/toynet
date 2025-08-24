import numpy as np

from toynet.config import NUMERICAL_EPS
from toynet.functions.abstract import LossFunction


class BinaryCrossEntropy(LossFunction):
    def __init__(self) -> None:
        self._fn = self._num_safe_bce
        self._fn_derivative = self._num_safe_bce_deriv

    def _num_safe_bce(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, NUMERICAL_EPS, 1 - NUMERICAL_EPS)
        return -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))  # type: ignore[no-any-return]

    def _num_safe_bce_deriv(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, NUMERICAL_EPS, 1 - NUMERICAL_EPS)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)  # type: ignore[no-any-return]


class MeanSquaredError(LossFunction):
    def __init__(self) -> None:
        self._fn = lambda y_true, y_pred: np.array([np.mean((y_true - y_pred) ** 2)]).reshape(-1, 1)
        self._fn_derivative = lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size


class CategoricalCrossEntropy(LossFunction):
    def __init__(self) -> None:
        self._fn = self._num_safe_cce
        self._fn_derivative = self._num_safe_cce_derivative

    def _num_safe_cce(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, NUMERICAL_EPS, 1 - NUMERICAL_EPS)
        return -np.sum(y_true * np.log(y_pred_clipped), axis=-1).reshape(-1, 1)  # type: ignore[no-any-return]

    def _num_safe_cce_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true  # type: ignore[no-any-return]
