import logging
from typing import Protocol

import numpy as np

from toynet.data_loaders.protocol import DataLoader
from toynet.functions.abstract import LossFunction
from toynet.layers.protocol import Layer
from toynet.optimizers.protocol import Optimizer
from toynet.policies.protocol import Policy


class NeuralNetwork(Protocol):
    optimizer: Optimizer
    logger: logging.Logger

    def __init__(
        self, layers: list[Layer], loss_function: LossFunction, optimizer: Optimizer
    ) -> None: ...

    def forward(self, network_input: np.ndarray) -> np.ndarray: ...

    def backprop(self, network_input: np.ndarray, y_true: np.ndarray) -> np.ndarray: ...

    def train(
        self, data_loader: DataLoader, epochs: int, policies: list[Policy] | None = None
    ) -> None: ...

    def save_to_npz(self, filename: str) -> None: ...

    def load_from_npz(self, filename: str) -> None: ...

    def __call__(self, network_input: np.ndarray) -> np.ndarray: ...
