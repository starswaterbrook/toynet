from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from toynet.data_loaders import BasicDataLoader
from toynet.functions import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Identity,
    MeanSquaredError,
    ReLU,
    Sigmoid,
    Softmax,
)
from toynet.layers import Dense
from toynet.networks import MultiLayerPerceptron
from toynet.optimizers import Adam, BasicOptimizer
from toynet.policies.reduce_lr_on_plateu import ReduceLROnPlateau
from toynet.policies.save_best_model import SaveBestModel
from toynet.policies.validation_early_stop import ValidationLossEarlyStop


class TestMultiLayerPerceptron:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)
        self.xor_X = np.array(
            [
                np.array([0.0, 1.0]).reshape(1, -1),
                np.array([0.0, 0.0]).reshape(1, -1),
                np.array([1.0, 0.0]).reshape(1, -1),
                np.array([1.0, 1.0]).reshape(1, -1),
            ]
        )
        self.xor_y = np.array(
            [
                np.array([1.0]).reshape(1, -1),
                np.array([0.0]).reshape(1, -1),
                np.array([1.0]).reshape(1, -1),
                np.array([0.0]).reshape(1, -1),
            ]
        )

        self.regression_X = np.array(
            [
                np.array([1.0, 2.0, 3.0]).reshape(1, -1),
                np.array([0.0, 1.0, 2.0]).reshape(1, -1),
                np.array([2.0, 1.0, 1.0]).reshape(1, -1),
                np.array([3.0, 0.0, 2.0]).reshape(1, -1),
                np.array([1.0, 1.0, 1.0]).reshape(1, -1),
                np.array([2.0, 2.0, 2.0]).reshape(1, -1),
                np.array([0.0, 0.0, 1.0]).reshape(1, -1),
                np.array([1.0, 0.0, 0.0]).reshape(1, -1),
                np.array([0.5, 0.5, 1.0]).reshape(1, -1),
                np.array([1.5, 1.5, 2.0]).reshape(1, -1),
            ]
        )
        self.regression_y = np.array(
            [
                np.array([6.0]).reshape(1, -1),
                np.array([3.0]).reshape(1, -1),
                np.array([4.0]).reshape(1, -1),
                np.array([5.0]).reshape(1, -1),
                np.array([3.0]).reshape(1, -1),
                np.array([6.0]).reshape(1, -1),
                np.array([1.0]).reshape(1, -1),
                np.array([1.0]).reshape(1, -1),
                np.array([2.0]).reshape(1, -1),
                np.array([5.0]).reshape(1, -1),
            ]
        )

        self.multi_class_X = np.array(
            [
                np.array([0.0, 1.0]).reshape(1, -1),
                np.array([0.0, 0.0]).reshape(1, -1),
                np.array([1.0, 0.0]).reshape(1, -1),
                np.array([1.0, 1.0]).reshape(1, -1),
            ]
        )
        self.multi_class_y = np.array(
            [
                np.array([1.0, 0.0, 0.0]).reshape(1, -1),
                np.array([0.0, 0.0, 1.0]).reshape(1, -1),
                np.array([1.0, 0.0, 0.0]).reshape(1, -1),
                np.array([0.0, 0.0, 1.0]).reshape(1, -1),
            ]
        )

    def test_basic_xor_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 32, ReLU),
                Dense(32, 16, ReLU),
                Dense(16, 1, Sigmoid),
            ],
            loss_function=BinaryCrossEntropy,
            optimizer=BasicOptimizer(learning_rate=0.01),
        )
        data_loader = BasicDataLoader(self.xor_X, self.xor_y, batch_size=2, validation_split=0.0)
        nnet.train(data_loader, epochs=1000)
        np.testing.assert_array_almost_equal(nnet(np.array([0.0, 1.0]))[0], [1.0], decimal=1)

    def test_advanced_xor_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 32, ReLU),
                Dense(32, 16, ReLU),
                Dense(16, 1, Sigmoid),
            ],
            loss_function=BinaryCrossEntropy,
            optimizer=Adam(learning_rate=0.01),
        )
        nnet.save_to_npz = MagicMock()
        data_loader = BasicDataLoader(self.xor_X, self.xor_y, batch_size=2, validation_split=0.0)

        nnet.train(
            data_loader,
            epochs=1000,
            policies=[
                ValidationLossEarlyStop("", patience=10, should_save=False),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
                SaveBestModel("", save_grace_period=20),
            ],
        )

        np.testing.assert_array_almost_equal(nnet(np.array([0.0, 1.0]))[0], [1.0], decimal=1)

    def test_basic_multi_class_xor_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 32, ReLU),
                Dense(32, 16, ReLU),
                Dense(16, 3, Softmax),
            ],
            loss_function=CategoricalCrossEntropy,
            optimizer=BasicOptimizer(learning_rate=0.01),
        )
        nnet.save_to_npz = MagicMock()
        data_loader = BasicDataLoader(
            self.multi_class_X, self.multi_class_y, batch_size=2, validation_split=0.0
        )
        nnet.train(data_loader, epochs=1000)
        np.testing.assert_array_almost_equal(
            nnet(np.array([0.0, 1.0])), np.array([[0.9, 0.005, 0.005]]), decimal=1
        )
        np.testing.assert_array_almost_equal(
            nnet(np.array([1.0, 1.0])), np.array([[0.005, 0.005, 0.9]]), decimal=1
        )

    def test_advanced_multi_class_xor_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 32, ReLU),
                Dense(32, 16, ReLU),
                Dense(16, 3, Softmax),
            ],
            loss_function=CategoricalCrossEntropy,
            optimizer=Adam(learning_rate=0.01),
        )
        nnet.save_to_npz = MagicMock()
        data_loader = BasicDataLoader(
            self.multi_class_X, self.multi_class_y, batch_size=2, validation_split=0.0
        )
        nnet.train(
            data_loader,
            epochs=1000,
            policies=[
                ValidationLossEarlyStop("", patience=10, should_save=False),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
                SaveBestModel("", save_grace_period=20),
            ],
        )
        np.testing.assert_array_almost_equal(
            nnet(np.array([0.0, 1.0])), np.array([[1.0, 0.0, 0.0]]), decimal=2
        )
        np.testing.assert_array_almost_equal(
            nnet(np.array([1.0, 1.0])), np.array([[0.0, 0.0, 1.0]]), decimal=2
        )

    def test_basic_regression_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(3, 64, ReLU),
                Dense(64, 32, ReLU),
                Dense(32, 1, Identity),
            ],
            loss_function=MeanSquaredError,
            optimizer=BasicOptimizer(learning_rate=0.01),
        )
        data_loader = BasicDataLoader(
            self.regression_X, self.regression_y, batch_size=2, validation_split=0.0
        )
        nnet.train(data_loader, epochs=1000)
        np.testing.assert_array_almost_equal(nnet(np.array([0.0, 1.0, 2.0]))[0], [3.0], decimal=1)

    def test_advanced_regression_mlp(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(3, 64, ReLU),
                Dense(64, 32, ReLU),
                Dense(32, 1, Identity),
            ],
            loss_function=MeanSquaredError,
            optimizer=Adam(learning_rate=0.01),
        )
        data_loader = BasicDataLoader(
            self.regression_X, self.regression_y, batch_size=2, validation_split=0.0
        )
        nnet.save_to_npz = MagicMock()
        nnet.train(
            data_loader,
            epochs=1000,
            policies=[
                ValidationLossEarlyStop("", patience=10, should_save=False),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
                SaveBestModel("", save_grace_period=20),
            ],
        )
        np.testing.assert_array_almost_equal(nnet(np.array([0.0, 1.0, 2.0]))[0], [3.0], decimal=1)
        np.testing.assert_array_almost_equal(nnet(np.array([3.0, 4.0, 5.0]))[0], [12.0], decimal=1)

    def test_basic_regression_mlp_with_validation(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(3, 64, ReLU),
                Dense(64, 32, ReLU),
                Dense(32, 1, Identity),
            ],
            loss_function=MeanSquaredError,
            optimizer=BasicOptimizer(learning_rate=0.01),
        )
        data_loader = BasicDataLoader(
            self.regression_X, self.regression_y, batch_size=2, validation_split=0.2
        )
        nnet.train(data_loader, epochs=1000)
        np.testing.assert_array_almost_equal(nnet(np.array([0.0, 1.0, 2.0]))[0], [3.0], decimal=1)

    def test_advanced_regression_mlp_with_validation(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(3, 64, ReLU),
                Dense(64, 32, ReLU),
                Dense(32, 1, Identity),
            ],
            loss_function=MeanSquaredError,
            optimizer=Adam(learning_rate=0.01),
        )
        data_loader = BasicDataLoader(
            self.regression_X, self.regression_y, batch_size=2, validation_split=0.2
        )
        nnet.save_to_npz = MagicMock()
        nnet.train(
            data_loader,
            epochs=1000,
            policies=[
                ValidationLossEarlyStop("", patience=10, should_save=False),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
                SaveBestModel("", save_grace_period=20),
            ],
        )
        np.testing.assert_array_almost_equal(nnet(np.array([1.0, 1.0, 2.0]))[0], [4.0], decimal=1)
        np.testing.assert_array_almost_equal(nnet(np.array([0.5, 0.5, 1.0]))[0], [2.0], decimal=1)

    def test_load_model(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 32, ReLU),
                Dense(32, 16, ReLU),
                Dense(16, 3, Softmax),
            ],
            loss_function=CategoricalCrossEntropy,
            optimizer=Adam(learning_rate=0.01),
        )

        test_file_dir = Path(__file__).parent
        model_path = test_file_dir / "models" / "multiclass_xor.testmodel"

        nnet.load_from_npz(str(model_path))

        np.testing.assert_array_almost_equal(
            nnet(np.array([0.0, 1.0])), np.array([[1.0, 0.0, 0.0]]), decimal=2
        )
        np.testing.assert_array_almost_equal(
            nnet(np.array([1.0, 1.0])), np.array([[0.0, 0.0, 1.0]]), decimal=2
        )

    def test_save_model(self) -> None:
        nnet = MultiLayerPerceptron(
            [
                Dense(2, 4, ReLU),
                Dense(4, 3, Softmax),
            ],
            loss_function=CategoricalCrossEntropy,
            optimizer=Adam(learning_rate=0.01),
        )

        with patch("numpy.savez") as mock_savez:
            nnet.save_to_npz("test_model.npz")

            mock_savez.assert_called_once()

            call_args = mock_savez.call_args
            assert call_args[0][0] == "test_model.npz"

            call_kwargs = call_args[1]

            expected_keys = {"W0", "b0", "W1", "b1"}
            assert set(call_kwargs.keys()) == expected_keys

            assert call_kwargs["W0"].shape == (2, 4)
            assert call_kwargs["b0"].shape == (1, 4)

            assert call_kwargs["W1"].shape == (4, 3)
            assert call_kwargs["b1"].shape == (1, 3)

            for key, value in call_kwargs.items():
                assert isinstance(value, np.ndarray), (
                    f"Expected numpy array for {key}, got {type(value)}"
                )
