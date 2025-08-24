from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pandas as pd

from toynet.data_loaders.protocol import DataLoader


class CSVDataLoader(DataLoader):
    def __init__(  # noqa: PLR0913
        self,
        csv_file: str,
        label_cols: list[str],
        batch_size: int = 32,
        validation_split: float = 0.0,
        shuffle: bool = True,
        transform: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.label_cols = label_cols
        self.shuffle = shuffle
        self.transform = transform

        self.columns = pd.read_csv(csv_file, nrows=0).columns.tolist()

        with Path.open(Path(csv_file), mode="rb") as f:
            self.n_samples = sum(1 for _ in f) - 1
        self.n_val = int(self.n_samples * validation_split)
        self.n_train = self.n_samples - self.n_val

    def _get_reader(self, start_idx: int = 0, end_idx: int | None = None) -> pd.DataFrame:
        nrows = None if end_idx is None else end_idx - start_idx
        skiprows = range(1, start_idx + 1)
        return pd.read_csv(self.csv_file, skiprows=skiprows, nrows=nrows)

    def _batch_generator(
        self, start_idx: int, end_idx: int
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        reader = self._get_reader(start_idx, end_idx)
        if self.shuffle:
            reader = reader.sample(frac=1).reset_index(drop=True)

        y = reader[self.label_cols].to_numpy(dtype=np.float32)
        X = []
        for _, row in reader.iterrows():
            X.append(row.drop(labels=self.label_cols).to_numpy(dtype=np.float32).reshape(1, -1))

        X_array = np.array(X)
        if self.transform:
            X_array, y = self.transform(X_array, y)

        yield X_array, y

    def train_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        return self._batch_generator(0, self.n_train)

    def val_generator(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if self.n_val == 0:
            return iter(())  # type: ignore[return-value]
        return self._batch_generator(self.n_train, self.n_samples)
