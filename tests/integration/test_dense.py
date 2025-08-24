import numpy as np
import pytest

from toynet.functions.activation import Identity, ReLU, Sigmoid
from toynet.initalizers.he import He
from toynet.initalizers.xavier import Xavier
from toynet.layers.dense import Dense


class TestDenseLayer:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)
        self.input_size = 4
        self.output_size = 2

        self.layer_relu = Dense(self.input_size, self.output_size, ReLU)
        self.layer_sigmoid = Dense(self.input_size, self.output_size, Sigmoid)
        self.layer_identity = Dense(self.input_size, self.output_size, Identity)

        self.simple_layer = Dense(2, 1, Identity)

    def test_initialization(self) -> None:
        assert self.layer_relu._weights.shape == (self.input_size, self.output_size)
        assert self.layer_relu._bias.shape == (1, self.output_size)

        assert self.layer_relu.grad_w_accum.shape == (self.input_size, self.output_size)
        assert self.layer_relu.grad_b_accum.shape == (1, self.output_size)

        np.testing.assert_array_equal(
            self.layer_relu.grad_w_accum, np.zeros_like(self.layer_relu._weights)
        )
        np.testing.assert_array_equal(
            self.layer_relu.grad_b_accum, np.zeros_like(self.layer_relu._bias)
        )
        assert self.layer_relu.activation_fn is not None

    def test_different_initializers(self) -> None:
        layer_he = Dense(self.input_size, self.output_size, ReLU, He)
        layer_xavier = Dense(self.input_size, self.output_size, ReLU, Xavier)

        layer_he_expected_weights = np.array(
            [[-0.087, -0.662], [-0.189, 0.375], [-0.489, -0.281], [-0.486, -0.598]]
        )

        layer_xavier_expected_weights = np.array(
            [[-0.388, -0.007], [-0.645, 0.135], [0.958, 0.428], [-0.111, -0.512]]
        )
        np.testing.assert_array_almost_equal(
            layer_he._weights, layer_he_expected_weights, decimal=2
        )
        np.testing.assert_array_almost_equal(
            layer_xavier._weights, layer_xavier_expected_weights, decimal=2
        )

        assert layer_he._weights.shape == (self.input_size, self.output_size)
        assert layer_xavier._weights.shape == (self.input_size, self.output_size)

    def test_forward_pass_calculation_with_activation(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        output = self.layer_relu.forward(layer_input, apply_activation=True)

        assert output.shape == (1, self.output_size)

        z_expected = layer_input @ self.layer_relu._weights + self.layer_relu._bias
        h_expected = np.maximum(0, z_expected)

        np.testing.assert_array_almost_equal(output, h_expected)
        np.testing.assert_array_almost_equal(self.layer_relu.z, z_expected)
        np.testing.assert_array_almost_equal(self.layer_relu.h, h_expected)
        np.testing.assert_array_equal(self.layer_relu.last_input, layer_input)

    def test_forward_pass_calculation_without_activation(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        output = self.layer_relu.forward(layer_input, apply_activation=False)

        assert output.shape == (1, self.output_size)
        z_expected = layer_input @ self.layer_relu._weights + self.layer_relu._bias

        np.testing.assert_array_almost_equal(output, z_expected)
        np.testing.assert_array_almost_equal(self.layer_relu.h, z_expected)

    def test_forward_batch_input(self) -> None:
        layer_batch_input = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        output = self.layer_identity.forward(layer_batch_input)

        assert output.shape == (4, self.output_size)
        expected_output = np.array(
            [
                [-0.121925, -0.62074],
                [0.02985, 0.412113],
                [-0.778255, 0.809442],
                [0.637521, 0.355317],
            ]
        )
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_backprop_calculation_identity_activation(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.layer_identity.forward(layer_input)
        grad_input = np.array([[1.0, -0.5]])
        grad_output = self.layer_identity.backpass(grad_input)

        assert grad_output.shape == layer_input.shape

        grad_z_expected = grad_input
        grad_w_expected = layer_input.T @ grad_z_expected
        grad_output_expected = grad_z_expected @ self.layer_identity._weights.T

        np.testing.assert_array_almost_equal(grad_output, grad_output_expected)
        np.testing.assert_array_almost_equal(self.layer_identity.grad_w_accum, grad_w_expected)
        np.testing.assert_array_almost_equal(self.layer_identity.grad_b_accum, grad_z_expected)

    def test_backprop_calculation_relu_activation(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.layer_relu.forward(layer_input)

        grad_input = np.array([[1.0, 1.0]])
        self.layer_relu.backpass(grad_input)

        relu_derivative = np.where(self.layer_relu.z > 0, 1.0, 0.0)
        grad_z_expected = grad_input * relu_derivative

        grad_w_expected = layer_input.T @ grad_z_expected
        np.testing.assert_array_almost_equal(self.layer_relu.grad_w_accum, grad_w_expected)

    def test_gradient_accumulation(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        grad_input = np.array([[1.0, 1.0]])

        self.layer_identity.forward(layer_input)
        self.layer_identity.backpass(grad_input)

        grad_w_after_first = self.layer_identity.grad_w_accum.copy()
        grad_b_after_first = self.layer_identity.grad_b_accum.copy()

        self.layer_identity.forward(layer_input)
        self.layer_identity.backpass(grad_input)

        expected_grad_w = 2 * grad_w_after_first
        expected_grad_b = 2 * grad_b_after_first

        np.testing.assert_array_almost_equal(self.layer_identity.grad_w_accum, expected_grad_w)
        np.testing.assert_array_almost_equal(self.layer_identity.grad_b_accum, expected_grad_b)

    def test_reset_accumulated_gradient(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.layer_identity.forward(layer_input)
        self.layer_identity.backpass(np.array([[1.0, 1.0]]))

        assert not np.all(self.layer_identity.grad_w_accum == 0)
        assert not np.all(self.layer_identity.grad_b_accum == 0)

        self.layer_identity.reset_accumulated_gradient()

        np.testing.assert_array_equal(
            self.layer_identity.grad_w_accum,
            np.zeros_like(self.layer_identity._weights),
        )
        np.testing.assert_array_equal(
            self.layer_identity.grad_b_accum, np.zeros_like(self.layer_identity._bias)
        )

    def test_parameters_property(self) -> None:
        params = self.layer_relu.parameters

        assert len(params) == 2
        assert np.shares_memory(params[0], self.layer_relu._weights)
        assert np.shares_memory(params[1], self.layer_relu._bias)

    def test_gradients_property(self) -> None:
        grads = self.layer_relu.gradients

        assert len(grads) == 2
        assert np.shares_memory(grads[0], self.layer_relu.grad_w_accum)
        assert np.shares_memory(grads[1], self.layer_relu.grad_b_accum)

    def test_minimum_dimensions(self) -> None:
        layer = Dense(1, 1, Identity)
        input_data = np.array([[5.0]])

        output = layer.forward(input_data)
        assert output.shape == (1, 1)

        layer.backpass(np.array([[1.0]]))
        assert layer.grad_w_accum.shape == (1, 1)

    def test_different_activation_functions(self) -> None:
        layer_input = np.array([[1.0, 0.0, 0.0, 0.0]])
        activations = [ReLU, Sigmoid, Identity]
        expected_outputs = {
            ReLU: np.array([[0.0, 0.0]]),
            Sigmoid: np.array([[0.384, 0.498]]),
            Identity: np.array([[-0.528, 1.197]]),
        }
        expected_grad_outputs = {
            ReLU: np.array([[0.0, 0.0, 0.0, 0.0]]),
            Sigmoid: np.array([[-0.114, -0.145, 0.409, -0.189]]),
            Identity: np.array([[0.668, -0.414, 1.620, 0.521]]),
        }

        for activation_class in activations:
            layer = Dense(self.input_size, self.output_size, activation_class)

            output = layer.forward(layer_input)
            assert output.shape == (1, self.output_size)
            np.testing.assert_allclose(output, expected_outputs[activation_class], atol=1e-2)

            grad_input = np.ones((1, self.output_size))
            grad_output = layer.backpass(grad_input)
            assert grad_output.shape == layer_input.shape
            np.testing.assert_allclose(
                grad_output, expected_grad_outputs[activation_class], atol=1e-2
            )
