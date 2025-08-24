import tempfile
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from toynet.data_loaders.basic_data_loader import BasicDataLoader
from toynet.data_loaders.csv_data_loader import CSVDataLoader


class TestDataLoaders:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)

        self.X = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]
        )
        self.y = np.array([0, 1, 0, 1, 0, 1])

        self.batch_size = 2
        self.validation_split = 0.2

        self.csv_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature3": [10.0, 20.0, 30.0, 40.0, 50.0],
                "label1": [0, 1, 0, 1, 0],
                "label2": [1, 0, 1, 0, 1],
            }
        )

        self.temp_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.csv_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

    def test_basic_data_loader_initialization(self) -> None:
        loader = BasicDataLoader(
            self.X,
            self.y,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
        )

        assert loader.batch_size == self.batch_size
        assert loader.validation_split == self.validation_split
        assert loader.n_samples == len(self.X)
        assert loader.n_val == int(len(self.X) * self.validation_split)
        assert loader.n_train == len(self.X) - loader.n_val

        np.testing.assert_array_equal(loader.X, self.X)
        np.testing.assert_array_equal(loader.y, self.y)

    def test_basic_data_loader_train_generator(self) -> None:
        loader = BasicDataLoader(self.X, self.y, validation_split=self.validation_split)

        train_gen = loader.train_generator()
        assert isinstance(train_gen, Generator)

        X_batch, y_batch = next(train_gen)

        expected_n_train = len(self.X) - int(len(self.X) * self.validation_split)
        assert len(X_batch) == expected_n_train
        assert len(y_batch) == expected_n_train

        assert X_batch.shape[1] == 1
        assert X_batch.shape[2] == self.X.shape[1]

    def test_basic_data_loader_val_generator(self) -> None:
        loader = BasicDataLoader(self.X, self.y, validation_split=self.validation_split)

        val_gen = loader.val_generator()
        assert isinstance(val_gen, Generator)

        X_batch, y_batch = next(val_gen)

        expected_n_val = int(len(self.X) * self.validation_split)
        assert len(X_batch) == expected_n_val
        assert len(y_batch) == expected_n_val

    def test_basic_data_loader_no_validation_split(self) -> None:
        loader = BasicDataLoader(self.X, self.y, validation_split=0.0)

        assert loader.n_val == 0
        assert loader.n_train == len(self.X)

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)
        assert len(X_batch) == len(self.X)
        assert len(y_batch) == len(self.X)

        val_gen = loader.val_generator()
        X_batch, y_batch = next(val_gen)
        assert len(X_batch) == 0
        assert len(y_batch) == 0

    def test_basic_data_loader_full_validation_split(self) -> None:
        loader = BasicDataLoader(self.X, self.y, validation_split=1.0)

        assert loader.n_val == len(self.X)
        assert loader.n_train == 0

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)
        assert len(X_batch) == 0
        assert len(y_batch) == 0

        val_gen = loader.val_generator()
        X_batch, y_batch = next(val_gen)
        assert len(X_batch) == len(self.X)
        assert len(y_batch) == len(self.X)

    def test_csv_data_loader_initialization(self) -> None:
        label_cols = ["label1", "label2"]
        loader = CSVDataLoader(
            self.temp_csv.name,
            label_cols=label_cols,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
        )

        assert loader.csv_file == self.temp_csv.name
        assert loader.label_cols == label_cols
        assert loader.batch_size == self.batch_size
        assert loader.n_samples == len(self.csv_data)
        assert loader.n_val == int(len(self.csv_data) * self.validation_split)
        assert loader.n_train == len(self.csv_data) - loader.n_val

    def test_csv_data_loader_columns_detection(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        expected_columns = ["feature1", "feature2", "feature3", "label1", "label2"]
        assert loader.columns == expected_columns

    def test_csv_data_loader_train_generator(self) -> None:
        label_cols = ["label1", "label2"]
        loader = CSVDataLoader(
            self.temp_csv.name,
            label_cols=label_cols,
            validation_split=self.validation_split,
        )

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)

        expected_n_train = len(self.csv_data) - int(len(self.csv_data) * self.validation_split)
        assert len(X_batch) == expected_n_train
        assert len(y_batch) == expected_n_train
        assert y_batch.shape[1] == len(label_cols)

    def test_csv_data_loader_val_generator(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(
            self.temp_csv.name,
            label_cols=label_cols,
            validation_split=self.validation_split,
        )

        val_gen = loader.val_generator()
        X_batch, y_batch = next(val_gen)

        expected_n_val = int(len(self.csv_data) * self.validation_split)
        assert len(X_batch) == expected_n_val
        assert len(y_batch) == expected_n_val

    def test_csv_data_loader_no_validation(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols, validation_split=0.0)

        assert loader.n_val == 0

        val_gen = loader.val_generator()
        with pytest.raises(StopIteration):
            next(val_gen)

    def test_csv_data_loader_feature_label_separation(self) -> None:
        label_cols = ["label1", "label2"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols, shuffle=False)

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)

        feature_cols = [col for col in self.csv_data.columns if col not in label_cols]
        expected_feature_count = len(feature_cols)

        assert X_batch[0].shape[1] == expected_feature_count
        assert y_batch.shape[1] == len(label_cols)

    def test_csv_data_loader_transform_function(self) -> None:
        def dummy_transform(X, y):
            X_transformed = [x * 2 for x in X]
            y_transformed = y * 3
            return X_transformed, y_transformed

        label_cols = ["label1"]
        loader = CSVDataLoader(
            self.temp_csv.name,
            label_cols=label_cols,
            transform=dummy_transform,
            shuffle=False,
        )

        train_gen = loader.train_generator()
        _, y_batch = next(train_gen)

        expected_y_first = self.csv_data["label1"].iloc[0] * 3
        assert y_batch[0, 0] == expected_y_first

    def test_csv_data_loader_shuffle_behavior(self) -> None:
        label_cols = ["label1"]

        loader_no_shuffle = CSVDataLoader(self.temp_csv.name, label_cols=label_cols, shuffle=False)
        loader_shuffle = CSVDataLoader(self.temp_csv.name, label_cols=label_cols, shuffle=True)

        train_gen_no_shuffle = loader_no_shuffle.train_generator()
        X_no_shuffle, _ = next(train_gen_no_shuffle)

        train_gen_shuffle = loader_shuffle.train_generator()
        X_shuffle, _ = next(train_gen_shuffle)

        first_row_no_shuffle = X_no_shuffle[0].flatten()
        _ = X_shuffle[0].flatten()

        expected_first_row = np.array([1.0, 0.1, 10.0, 1.0])
        np.testing.assert_allclose(first_row_no_shuffle, expected_first_row, atol=1e-6)

    def test_data_loader_interface_consistency(self) -> None:
        basic_loader = BasicDataLoader(self.X, self.y)
        csv_loader = CSVDataLoader(self.temp_csv.name, label_cols=["label1"])

        loaders = [basic_loader, csv_loader]

        for loader in loaders:
            assert hasattr(loader, "batch_size")
            assert hasattr(loader, "train_generator")
            assert hasattr(loader, "val_generator")
            assert callable(loader.train_generator)
            assert callable(loader.val_generator)

    def test_basic_data_loader_data_types(self) -> None:
        loader = BasicDataLoader(self.X, self.y)

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)

        assert isinstance(X_batch, np.ndarray)
        assert isinstance(y_batch, np.ndarray)

    def test_csv_data_loader_data_types(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        train_gen = loader.train_generator()
        X_batch, y_batch = next(train_gen)

        assert isinstance(X_batch, np.ndarray)
        assert isinstance(y_batch, np.ndarray)
        assert y_batch.dtype == np.float32

    def test_basic_data_loader_empty_data(self) -> None:
        empty_X = np.array([]).reshape(0, 3)
        empty_y = np.array([])

        loader = BasicDataLoader(empty_X, empty_y)

        assert loader.n_samples == 0
        assert loader.n_train == 0
        assert loader.n_val == 0

    def test_csv_data_loader_single_label_column(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        train_gen = loader.train_generator()
        _, y_batch = next(train_gen)

        assert y_batch.shape[1] == 1

    def test_csv_data_loader_multiple_label_columns(self) -> None:
        label_cols = ["label1", "label2"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        train_gen = loader.train_generator()
        _, y_batch = next(train_gen)

        assert y_batch.shape[1] == 2

    def test_basic_data_loader_different_validation_splits(self) -> None:
        splits = [0.0, 0.1, 0.2, 0.5, 1.0]

        for split in splits:
            loader = BasicDataLoader(self.X, self.y, validation_split=split)
            expected_val = int(len(self.X) * split)
            expected_train = len(self.X) - expected_val

            assert loader.n_val == expected_val
            assert loader.n_train == expected_train

    def test_csv_data_loader_get_reader_functionality(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        reader_full = loader._get_reader()
        assert len(reader_full) == len(self.csv_data)

        reader_partial = loader._get_reader(start_idx=1, end_idx=3)
        assert len(reader_partial) == 2

    def test_basic_data_loader_generator_exhaustion(self) -> None:
        loader = BasicDataLoader(self.X, self.y)

        train_gen = loader.train_generator()
        next(train_gen)

        with pytest.raises(StopIteration):
            next(train_gen)

    def test_csv_data_loader_generator_exhaustion(self) -> None:
        label_cols = ["label1"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols)

        train_gen = loader.train_generator()
        next(train_gen)

        with pytest.raises(StopIteration):
            next(train_gen)

    def test_data_shape_consistency(self) -> None:
        basic_loader = BasicDataLoader(self.X, self.y, validation_split=0.2)

        train_gen = basic_loader.train_generator()
        X_train, y_train = next(train_gen)

        val_gen = basic_loader.val_generator()
        X_val, y_val = next(val_gen)

        assert X_train.shape[2] == X_val.shape[2]
        assert len(X_train) + len(X_val) == len(self.X)
        assert len(y_train) + len(y_val) == len(self.y)

    def test_csv_data_loader_feature_extraction(self) -> None:
        label_cols = ["label1", "label2"]
        loader = CSVDataLoader(self.temp_csv.name, label_cols=label_cols, shuffle=False)

        train_gen = loader.train_generator()
        X_batch, _ = next(train_gen)

        expected_features = self.csv_data.drop(columns=label_cols).iloc[0].to_numpy()
        actual_features = X_batch[0].flatten()

        np.testing.assert_allclose(actual_features, expected_features, atol=1e-6)

    def test_basic_data_loader_sample_indexing(self) -> None:
        loader = BasicDataLoader(self.X, self.y, validation_split=0.5)

        train_gen = loader.train_generator()
        X_train, _ = next(train_gen)

        val_gen = loader.val_generator()
        X_val, _ = next(val_gen)

        assert len(X_train) == loader.n_train
        assert len(X_val) == loader.n_val
        assert len(X_train) + len(X_val) == len(self.X)
