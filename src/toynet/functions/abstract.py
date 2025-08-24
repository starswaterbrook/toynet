from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class ActivationFunction(ABC):
    _fn: Callable[[np.ndarray], np.ndarray]
    _fn_derivative: Callable[[np.ndarray], np.ndarray]

    @abstractmethod
    def __init__(self) -> None: ...

    def __call__(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return self._fn_derivative(x)
        return self._fn(x)


class LossFunction(ABC):
    _fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _fn_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @abstractmethod
    def __init__(self) -> None: ...

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False
    ) -> np.ndarray:
        if derivative:
            return self._fn_derivative(y_true, y_pred)
        return self._fn(y_true, y_pred)
