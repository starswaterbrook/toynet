from collections.abc import Generator
from typing import Protocol

import numpy as np


class DataLoader(Protocol):
    n_val: int
    n_samples: int

    def __init__(self, batch_size: int = 32, validation_split: float = 0.0) -> None: ...

    def train_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]: ...

    def val_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]: ...
