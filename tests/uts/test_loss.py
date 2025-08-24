import numpy as np
import pytest

from toynet.config import NUMERICAL_EPS
from toynet.functions.loss import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
)


class TestLossFunctions:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)

        self.binary_y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        self.binary_y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])

        self.regression_y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.regression_y_pred = np.array([[1.1, 1.9], [2.8, 4.2], [5.2, 5.8]])

        self.categorical_y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.categorical_y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])

        self.perfect_pred = np.array([[1.0], [0.0], [1.0], [0.0]])
        self.worst_pred = np.array([[0.0], [1.0], [0.0], [1.0]])

        self.bce = BinaryCrossEntropy()
        self.mse = MeanSquaredError()
        self.cce = CategoricalCrossEntropy()

    def test_binary_cross_entropy_forward(self) -> None:
        loss = self.bce(self.binary_y_true, self.binary_y_pred)

        expected_loss = np.array([[-np.log(0.9)], [-np.log(0.9)], [-np.log(0.8)], [-np.log(0.8)]])

        np.testing.assert_allclose(loss, expected_loss, atol=1e-6)

    def test_binary_cross_entropy_derivative(self) -> None:
        derivative = self.bce(self.binary_y_true, self.binary_y_pred, derivative=True)

        expected_derivative = np.array(
            [
                [-(1.0 / 0.9) + (1.0 - 1.0) / (1.0 - 0.9)],
                [-(0.0 / 0.1) + (1.0 - 0.0) / (1.0 - 0.1)],
                [-(1.0 / 0.8) + (1.0 - 1.0) / (1.0 - 0.8)],
                [-(0.0 / 0.2) + (1.0 - 0.0) / (1.0 - 0.2)],
            ]
        )

        np.testing.assert_allclose(derivative, expected_derivative, atol=1e-6)

    def test_binary_cross_entropy_perfect_prediction(self) -> None:
        loss = self.bce(self.binary_y_true, self.perfect_pred)

        expected_loss = np.array(
            [
                [-np.log(1.0 - NUMERICAL_EPS)],
                [-np.log(1.0 - NUMERICAL_EPS)],
                [-np.log(1.0 - NUMERICAL_EPS)],
                [-np.log(1.0 - NUMERICAL_EPS)],
            ]
        )

        np.testing.assert_allclose(loss, expected_loss, atol=1e-10)

    def test_binary_cross_entropy_worst_prediction(self) -> None:
        loss = self.bce(self.binary_y_true, self.worst_pred)

        expected_loss = np.array(
            [
                [-np.log(NUMERICAL_EPS)],
                [-np.log(NUMERICAL_EPS)],
                [-np.log(NUMERICAL_EPS)],
                [-np.log(NUMERICAL_EPS)],
            ]
        )

        np.testing.assert_allclose(loss, expected_loss, atol=1e-4)

    def test_binary_cross_entropy_clipping(self) -> None:
        extreme_pred = np.array([[1e10], [-1e10], [0.0], [1.0]])

        loss = self.bce(self.binary_y_true, extreme_pred)

        assert not np.any(np.isnan(loss))
        assert not np.any(np.isinf(loss))

    def test_mean_squared_error_forward(self) -> None:
        loss = self.mse(self.regression_y_true, self.regression_y_pred)

        diff = self.regression_y_true - self.regression_y_pred
        expected_loss = np.mean(diff**2)

        np.testing.assert_allclose(loss, [[expected_loss]], atol=1e-10)

    def test_mean_squared_error_derivative(self) -> None:
        derivative = self.mse(self.regression_y_true, self.regression_y_pred, derivative=True)

        expected_derivative = (
            2 * (self.regression_y_pred - self.regression_y_true) / self.regression_y_true.size
        )

        np.testing.assert_allclose(derivative, expected_derivative, atol=1e-10)

    def test_mean_squared_error_perfect_prediction(self) -> None:
        perfect_regression_pred = self.regression_y_true.copy()
        loss = self.mse(self.regression_y_true, perfect_regression_pred)

        np.testing.assert_allclose(loss, [[0.0]], atol=1e-10)

    def test_mean_squared_error_zero_gradient_at_optimum(self) -> None:
        perfect_regression_pred = self.regression_y_true.copy()
        derivative = self.mse(self.regression_y_true, perfect_regression_pred, derivative=True)

        np.testing.assert_allclose(derivative, np.zeros_like(derivative), atol=1e-10)

    def test_categorical_cross_entropy_forward(self) -> None:
        loss = self.cce(self.categorical_y_true, self.categorical_y_pred)

        expected_loss = np.array([[-np.log(0.7)], [-np.log(0.8)], [-np.log(0.7)]])

        np.testing.assert_allclose(loss, expected_loss, atol=1e-10)

    def test_categorical_cross_entropy_derivative(self) -> None:
        derivative = self.cce(self.categorical_y_true, self.categorical_y_pred, derivative=True)

        expected_derivative = self.categorical_y_pred - self.categorical_y_true

        np.testing.assert_allclose(derivative, expected_derivative, atol=1e-10)

    def test_categorical_cross_entropy_perfect_prediction(self) -> None:
        perfect_categorical_pred = self.categorical_y_true.copy()
        perfect_categorical_pred = np.clip(
            perfect_categorical_pred, NUMERICAL_EPS, 1 - NUMERICAL_EPS
        )

        loss = self.cce(self.categorical_y_true, perfect_categorical_pred)

        expected_loss = np.array(
            [
                [-np.log(1.0 - NUMERICAL_EPS)],
                [-np.log(1.0 - NUMERICAL_EPS)],
                [-np.log(1.0 - NUMERICAL_EPS)],
            ]
        )

        np.testing.assert_allclose(loss, expected_loss, atol=1e-10)

    def test_categorical_cross_entropy_clipping(self) -> None:
        extreme_categorical_pred = np.array(
            [[1e10, -1e10, 0.0], [0.0, 1e10, -1e10], [-1e10, 0.0, 1e10]]
        )

        loss = self.cce(self.categorical_y_true, extreme_categorical_pred)

        assert not np.any(np.isnan(loss))
        assert not np.any(np.isinf(loss))

    def test_loss_function_interface_consistency(self) -> None:
        losses = [self.bce, self.mse, self.cce]
        test_pairs = [
            (self.binary_y_true, self.binary_y_pred),
            (self.regression_y_true, self.regression_y_pred),
            (self.categorical_y_true, self.categorical_y_pred),
        ]

        for loss_fn, (y_true, y_pred) in zip(losses, test_pairs, strict=False):
            forward_result = loss_fn(y_true, y_pred)
            assert isinstance(forward_result, np.ndarray)

            derivative_result = loss_fn(y_true, y_pred, derivative=True)
            assert isinstance(derivative_result, np.ndarray)
            assert derivative_result.shape == y_pred.shape

    def test_loss_function_array_shapes(self) -> None:
        test_cases = [
            (self.bce, self.binary_y_true, self.binary_y_pred),
            (self.mse, self.regression_y_true, self.regression_y_pred),
            (self.cce, self.categorical_y_true, self.categorical_y_pred),
        ]

        for loss_fn, y_true, y_pred in test_cases:
            forward_result = loss_fn(y_true, y_pred)
            derivative_result = loss_fn(y_true, y_pred, derivative=True)

            if loss_fn == self.mse:
                assert forward_result.shape == (1, 1)
            else:
                assert forward_result.shape == (y_true.shape[0], 1)

            assert derivative_result.shape == y_pred.shape

            batch_size = y_true.shape[0]
            feature_size = y_true.shape[1] if len(y_true.shape) > 1 else 1
            if loss_fn == self.mse:
                assert forward_result.ndim == 2
                assert len(forward_result) == 1
            else:
                assert forward_result.ndim == 2
                assert len(forward_result) == batch_size

            assert derivative_result.ndim == y_pred.ndim
            assert derivative_result.shape[0] == batch_size
            if len(y_pred.shape) > 1:
                assert derivative_result.shape[1] == feature_size

    def test_binary_cross_entropy_single_sample(self) -> None:
        single_true = np.array([[1.0]])
        single_pred = np.array([[0.8]])

        loss = self.bce(single_true, single_pred)
        expected = [-np.log(0.8)]

        np.testing.assert_allclose(loss, [expected], atol=1e-10)

    def test_mean_squared_error_single_sample(self) -> None:
        single_true = np.array([[2.0]])
        single_pred = np.array([[1.5]])

        loss = self.mse(single_true, single_pred)
        expected = (2.0 - 1.5) ** 2

        np.testing.assert_allclose(loss, [[expected]], atol=1e-10)

    def test_categorical_cross_entropy_single_sample(self) -> None:
        single_true = np.array([[1.0, 0.0, 0.0]])
        single_pred = np.array([[0.6, 0.3, 0.1]])

        loss = self.cce(single_true, single_pred)
        expected = [-np.log(0.6)]

        np.testing.assert_allclose(loss, [expected], atol=1e-10)

    def test_binary_cross_entropy_batch_properties(self) -> None:
        batch_loss = self.bce(self.binary_y_true, self.binary_y_pred)

        assert len(batch_loss) == len(self.binary_y_true)
        assert all(loss >= 0 for loss in batch_loss)

    def test_mean_squared_error_symmetry(self) -> None:
        loss1 = self.mse(self.regression_y_true, self.regression_y_pred)
        loss2 = self.mse(self.regression_y_pred, self.regression_y_true)

        np.testing.assert_allclose(loss1, loss2, atol=1e-10)

    def test_categorical_cross_entropy_batch_properties(self) -> None:
        batch_loss = self.cce(self.categorical_y_true, self.categorical_y_pred)

        assert len(batch_loss) == len(self.categorical_y_true)
        assert all(loss >= 0 for loss in batch_loss)

    def test_loss_functions_with_edge_case_inputs(self) -> None:
        edge_binary_true = np.array([[1.0], [0.0]])
        edge_binary_pred = np.array([[NUMERICAL_EPS], [1.0 - NUMERICAL_EPS]])

        bce_loss = self.bce(edge_binary_true, edge_binary_pred)
        assert not np.any(np.isnan(bce_loss))
        assert not np.any(np.isinf(bce_loss))

    def test_gradient_signs_correctness(self) -> None:
        derivative_bce = self.bce(self.binary_y_true, self.binary_y_pred, derivative=True)

        for i, (true_val, pred_val) in enumerate(
            zip(self.binary_y_true.flatten(), self.binary_y_pred.flatten(), strict=False)
        ):
            if true_val == 1.0 and pred_val < 1.0:
                assert derivative_bce[i, 0] < 0
            elif true_val == 0.0 and pred_val > 0.0:
                assert derivative_bce[i, 0] > 0

    def test_mse_derivative_direction(self) -> None:
        over_pred = self.regression_y_true + 0.5
        under_pred = self.regression_y_true - 0.5

        over_derivative = self.mse(self.regression_y_true, over_pred, derivative=True)
        under_derivative = self.mse(self.regression_y_true, under_pred, derivative=True)

        assert np.all(over_derivative > 0)
        assert np.all(under_derivative < 0)

    def test_categorical_cross_entropy_probability_distribution(self) -> None:
        uniform_pred = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])

        uniform_loss = self.cce(self.categorical_y_true, uniform_pred)
        confident_loss = self.cce(self.categorical_y_true, self.categorical_y_pred)

        assert np.all(uniform_loss > confident_loss)

    def test_all_losses_non_negative(self) -> None:
        bce_loss = self.bce(self.binary_y_true, self.binary_y_pred)
        mse_loss = self.mse(self.regression_y_true, self.regression_y_pred)
        cce_loss = self.cce(self.categorical_y_true, self.categorical_y_pred)

        assert np.all(bce_loss >= 0)
        assert np.all(mse_loss >= 0)
        assert np.all(cce_loss >= 0)
