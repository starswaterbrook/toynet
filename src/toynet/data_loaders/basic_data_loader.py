from collections.abc import Generator

import numpy as np

from toynet.data_loaders.protocol import DataLoader


class BasicDataLoader(DataLoader):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        validation_split: float = 0.0,
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.n_samples = len(X)
        self.n_val = int(self.n_samples * validation_split)
        self.n_train = self.n_samples - self.n_val

    def train_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        X = []
        y = []
        for i in range(self.n_train):
            X.append(self.X[i].reshape(1, -1))
            y.append(self.y[i])
        yield np.array(X), np.array(y)

    def val_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        X = []
        y = []
        for i in range(self.n_train, self.n_samples):
            X.append(self.X[i].reshape(1, -1))
            y.append(self.y[i])
        yield np.array(X), np.array(y)
