import numpy as np
import pytest

from toynet.config import NUMERICAL_EPS
from toynet.functions.activation import Identity, ReLU, Sigmoid, Softmax


class TestActivationFunctions:
    @pytest.fixture(autouse=True)
    def setup_method(self) -> None:
        np.random.seed(1)

        self.test_values = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        self.batch_values = np.array(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-5.0, -0.5, 0.5, 1.5, 5.0],
                [-10.0, -0.1, 0.1, 3.0, 10.0],
            ]
        )
        self.large_values = np.array([[1e99, -1e99, 1e46 + 2132142, -1e46 - 2132142]])
        self.softmax_values = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [10.0, 20.0, 30.0]])

        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.identity = Identity()
        self.softmax = Softmax()

    def test_relu_forward(self) -> None:
        output = self.relu(self.test_values)
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(output, expected)

        batch_output = self.relu(self.batch_values)
        expected_batch = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.5, 1.5, 5.0],
                [0.0, 0.0, 0.1, 3.0, 10.0],
            ]
        )
        np.testing.assert_array_equal(batch_output, expected_batch)

    def test_relu_derivative(self) -> None:
        derivative = self.relu(self.test_values, derivative=True)
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(derivative, expected)

        batch_derivative = self.relu(self.batch_values, derivative=True)
        expected_batch = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(batch_derivative, expected_batch)

    def test_sigmoid_forward(self) -> None:
        output = self.sigmoid(self.test_values)

        expected = np.array([[0.119, 0.269, 0.5, 0.731, 0.881]])
        np.testing.assert_allclose(output, expected, atol=1e-3)

        assert np.all(output >= NUMERICAL_EPS)
        assert np.all(output <= 1 - NUMERICAL_EPS)

    def test_sigmoid_derivative(self) -> None:
        derivative = self.sigmoid(self.test_values, derivative=True)
        assert np.all(derivative > 0)
        zero_idx = np.where(self.test_values == 0.0)
        if len(zero_idx[0]) > 0 and len(zero_idx[1]) > 0:
            zero_derivative = derivative[zero_idx[0][0], zero_idx[1][0]]
            assert zero_derivative == pytest.approx(0.25, abs=1e-3)

    def test_sigmoid_numerical_stability(self) -> None:
        output = self.sigmoid(self.large_values)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        assert np.all(output >= NUMERICAL_EPS)
        assert np.all(output <= 1 - NUMERICAL_EPS)

    def test_identity_forward(self) -> None:
        output = self.identity(self.test_values)
        np.testing.assert_array_equal(output, self.test_values)

        batch_output = self.identity(self.batch_values)
        np.testing.assert_array_equal(batch_output, self.batch_values)

    def test_identity_derivative(self) -> None:
        derivative = self.identity(self.test_values, derivative=True)
        expected = np.ones_like(self.test_values)
        np.testing.assert_array_equal(derivative, expected)

        batch_derivative = self.identity(self.batch_values, derivative=True)
        expected_batch = np.ones_like(self.batch_values)
        np.testing.assert_array_equal(batch_derivative, expected_batch)

    def test_softmax_forward(self) -> None:
        output = self.softmax(self.softmax_values)

        assert output.shape == self.softmax_values.shape

        row_sums = np.sum(output, axis=1)

        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-10)

        assert np.all(output > 0)
        assert np.all(output < 1)

    def test_softmax_numerical_stability(self) -> None:
        large_softmax_values = np.array(
            [[100.0, 101.0, 102.0], [1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]]
        )

        output = self.softmax(large_softmax_values)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        row_sums = np.sum(output, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-10)

    def test_softmax_derivative(self) -> None:
        derivative = self.softmax(self.softmax_values, derivative=True)

        assert derivative.shape == self.softmax_values.shape

        expected = np.ones_like(self.softmax_values)
        np.testing.assert_array_equal(derivative, expected)

    def test_softmax_properties(self) -> None:
        shift = 10.0
        shifted_values = self.softmax_values + shift

        original_output = self.softmax(self.softmax_values)
        shifted_output = self.softmax(shifted_values)

        np.testing.assert_allclose(original_output, shifted_output, atol=1e-10)

    def test_activation_function_interface(self) -> None:
        activations = [self.relu, self.sigmoid, self.identity, self.softmax]
        test_input = np.array([[1.0, -1.0, 0.0]])

        for activation in activations:
            output = activation(test_input)
            assert isinstance(output, np.ndarray)
            assert output.shape == test_input.shape

            derivative = activation(test_input, derivative=True)
            assert isinstance(derivative, np.ndarray)
            assert derivative.shape == test_input.shape

    def test_edge_cases_single_values(self) -> None:
        single_value = np.array([[0.0]])

        assert self.relu(single_value)[0, 0] == 0.0
        assert self.relu(single_value, derivative=True)[0, 0] == 0.0

        sigmoid_output = self.sigmoid(single_value)
        assert sigmoid_output[0, 0] == pytest.approx(0.5, abs=1e-10)

        assert self.identity(single_value)[0, 0] == 0.0
        assert self.identity(single_value, derivative=True)[0, 0] == 1.0

    def test_edge_cases_empty_arrays(self) -> None:
        empty_array = np.array([]).reshape(0, 1)

        for activation in [self.relu, self.sigmoid, self.identity]:
            output = activation(empty_array)
            assert output.shape == empty_array.shape

            derivative = activation(empty_array, derivative=True)
            assert derivative.shape == empty_array.shape

    def test_very_large_inputs(self) -> None:
        very_large = np.array([[1e10, -1e10, 1e15, -1e15]])

        relu_output = self.relu(very_large)
        expected = np.array([[1e10, 0.0, 1e15, 0.0]])
        np.testing.assert_array_equal(relu_output, expected)

        sigmoid_output = self.sigmoid(very_large)
        assert not np.any(np.isnan(sigmoid_output))
        assert not np.any(np.isinf(sigmoid_output))

        identity_output = self.identity(very_large)
        np.testing.assert_array_equal(identity_output, very_large)

    def test_activation_derivatives_at_critical_points(self) -> None:
        zero_point = np.array([[0.0]])
        relu_deriv = self.relu(zero_point, derivative=True)
        assert relu_deriv[0, 0] == 0.0

        sigmoid_deriv = self.sigmoid(zero_point, derivative=True)
        assert sigmoid_deriv[0, 0] == pytest.approx(0.25, abs=1e-10)

    def test_batch_consistency(self) -> None:
        individual_results = [self.relu(row.reshape(1, -1)) for row in self.batch_values]
        individual_combined = np.vstack(individual_results)

        batch_result = self.relu(self.batch_values)
        np.testing.assert_array_equal(batch_result, individual_combined)
