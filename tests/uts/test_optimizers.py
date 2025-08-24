import numpy as np
import pytest

from toynet.optimizers.adam import Adam
from toynet.optimizers.basic_optimizer import BasicOptimizer


class TestOptimizers:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)

        self.learning_rate = 0.01
        self.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.bias = np.array([[0.5, -0.5]])
        self.weight_gradients = np.array([[0.1, -0.2], [0.3, -0.1]])
        self.bias_gradients = np.array([[0.05, -0.1]])

        self.large_weights = np.random.randn(100, 50)
        self.large_bias = np.random.randn(1, 50)
        self.large_weight_grads = np.random.randn(100, 50)
        self.large_bias_grads = np.random.randn(1, 50)

        self.basic_optimizer = BasicOptimizer(learning_rate=self.learning_rate)
        self.adam_optimizer = Adam(learning_rate=self.learning_rate)

    def test_basic_optimizer_initialization(self) -> None:
        optimizer = BasicOptimizer(learning_rate=0.02)
        assert optimizer.learning_rate == 0.02

        default_optimizer = BasicOptimizer()
        assert default_optimizer.learning_rate == 0.01

    def test_adam_optimizer_initialization(self) -> None:
        optimizer = Adam(learning_rate=0.002, beta1=0.8, beta2=0.95, eps=1e-9)
        assert optimizer.learning_rate == 0.002
        assert optimizer.beta1 == 0.8
        assert optimizer.beta2 == 0.95
        assert optimizer.eps == 1e-9
        assert optimizer.t == 0
        assert len(optimizer.m) == 0
        assert len(optimizer.v) == 0

        default_optimizer = Adam()
        assert default_optimizer.learning_rate == 0.001
        assert default_optimizer.beta1 == 0.9
        assert default_optimizer.beta2 == 0.999
        assert default_optimizer.eps == 1e-8

    def test_basic_optimizer_single_step(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        expected_weights = self.weights - self.learning_rate * self.weight_gradients
        expected_bias = self.bias - self.learning_rate * self.bias_gradients

        self.basic_optimizer.step(params, grads)

        np.testing.assert_allclose(params[0], expected_weights, atol=1e-10)
        np.testing.assert_allclose(params[1], expected_bias, atol=1e-10)

    def test_basic_optimizer_multiple_steps(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        initial_weights = params[0].copy()
        initial_bias = params[1].copy()

        num_steps = 5
        for _ in range(num_steps):
            self.basic_optimizer.step(params, grads)

        expected_weights = initial_weights - num_steps * self.learning_rate * self.weight_gradients
        expected_bias = initial_bias - num_steps * self.learning_rate * self.bias_gradients

        np.testing.assert_allclose(params[0], expected_weights, atol=1e-10)
        np.testing.assert_allclose(params[1], expected_bias, atol=1e-10)

    def test_adam_optimizer_single_step(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        initial_weights = params[0].copy()
        initial_bias = params[1].copy()

        self.adam_optimizer.step(params, grads)

        assert self.adam_optimizer.t == 1
        assert len(self.adam_optimizer.m) == 2
        assert len(self.adam_optimizer.v) == 2

        assert not np.array_equal(params[0], initial_weights)
        assert not np.array_equal(params[1], initial_bias)

    def test_adam_optimizer_momentum_accumulation(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        self.adam_optimizer.step(params, grads)

        weight_key = id(params[0])
        bias_key = id(params[1])

        expected_m_weights = (1.0 - self.adam_optimizer.beta1) * self.weight_gradients
        expected_m_bias = (1.0 - self.adam_optimizer.beta1) * self.bias_gradients

        np.testing.assert_allclose(
            self.adam_optimizer.m[weight_key], expected_m_weights, atol=1e-10
        )
        np.testing.assert_allclose(self.adam_optimizer.m[bias_key], expected_m_bias, atol=1e-10)

    def test_adam_optimizer_variance_accumulation(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        self.adam_optimizer.step(params, grads)

        weight_key = id(params[0])
        bias_key = id(params[1])

        expected_v_weights = (1.0 - self.adam_optimizer.beta2) * (self.weight_gradients**2)
        expected_v_bias = (1.0 - self.adam_optimizer.beta2) * (self.bias_gradients**2)

        np.testing.assert_allclose(
            self.adam_optimizer.v[weight_key], expected_v_weights, atol=1e-10
        )
        np.testing.assert_allclose(self.adam_optimizer.v[bias_key], expected_v_bias, atol=1e-10)

    def test_adam_optimizer_bias_correction(self) -> None:
        grads = [self.weight_gradients, self.bias_gradients]

        steps = [1, 2, 5, 10]

        for target_step in steps:
            adam = Adam(learning_rate=self.learning_rate)
            test_params = [self.weights.copy(), self.bias.copy()]

            for _ in range(target_step):
                adam.step(test_params, grads)

            assert adam.t == target_step

            bias_correction_1 = 1.0 - adam.beta1**target_step
            bias_correction_2 = 1.0 - adam.beta2**target_step

            assert bias_correction_1 > 0
            assert bias_correction_2 > 0

    def test_optimizer_interface_consistency(self) -> None:
        optimizers = [self.basic_optimizer, self.adam_optimizer]

        for optimizer in optimizers:
            assert hasattr(optimizer, "learning_rate")
            assert hasattr(optimizer, "step")
            assert callable(optimizer.step)

    def test_basic_optimizer_zero_gradients(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        zero_grads = [np.zeros_like(self.weights), np.zeros_like(self.bias)]

        initial_weights = params[0].copy()
        initial_bias = params[1].copy()

        self.basic_optimizer.step(params, zero_grads)

        np.testing.assert_array_equal(params[0], initial_weights)
        np.testing.assert_array_equal(params[1], initial_bias)

    def test_adam_optimizer_zero_gradients(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        zero_grads = [np.zeros_like(self.weights), np.zeros_like(self.bias)]

        initial_weights = params[0].copy()
        initial_bias = params[1].copy()

        self.adam_optimizer.step(params, zero_grads)

        np.testing.assert_array_equal(params[0], initial_weights)
        np.testing.assert_array_equal(params[1], initial_bias)

    def test_adam_optimizer_none_gradients(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        none_grads = [None, self.bias_gradients]

        initial_weights = params[0].copy()
        initial_bias = params[1].copy()

        self.adam_optimizer.step(params, none_grads)

        np.testing.assert_array_equal(params[0], initial_weights)
        assert not np.array_equal(params[1], initial_bias)

    def test_adam_convergence_behavior(self) -> None:
        params = [np.array([[10.0]]), np.array([[5.0]])]
        target = [np.array([[0.0]]), np.array([[0.0]])]

        adam = Adam(learning_rate=0.1)

        initial_distance = np.sum((params[0] - target[0]) ** 2) + np.sum(
            (params[1] - target[1]) ** 2
        )

        for _ in range(1000):
            grads = [2 * (params[0] - target[0]), 2 * (params[1] - target[1])]
            adam.step(params, grads)

        final_distance = np.sum((params[0] - target[0]) ** 2) + np.sum((params[1] - target[1]) ** 2)

        assert final_distance < initial_distance
        assert final_distance < 0.1

    def test_optimizer_parameter_modification_inplace(self) -> None:
        params = [self.weights.copy(), self.bias.copy()]
        grads = [self.weight_gradients, self.bias_gradients]

        original_id_weights = id(params[0])
        original_id_bias = id(params[1])

        self.basic_optimizer.step(params, grads)

        assert id(params[0]) == original_id_weights
        assert id(params[1]) == original_id_bias

    def test_large_parameter_stability(self) -> None:
        params = [self.large_weights.copy(), self.large_bias.copy()]
        grads = [self.large_weight_grads, self.large_bias_grads]

        basic = BasicOptimizer(learning_rate=0.001)
        adam = Adam(learning_rate=0.001)

        basic_params = [p.copy() for p in params]
        adam_params = [p.copy() for p in params]

        basic.step(basic_params, grads)
        adam.step(adam_params, grads)

        assert not np.any(np.isnan(basic_params[0]))
        assert not np.any(np.isnan(basic_params[1]))
        assert not np.any(np.isnan(adam_params[0]))
        assert not np.any(np.isnan(adam_params[1]))

        assert not np.any(np.isinf(basic_params[0]))
        assert not np.any(np.isinf(basic_params[1]))
        assert not np.any(np.isinf(adam_params[0]))
        assert not np.any(np.isinf(adam_params[1]))

    def test_adam_epsilon_prevents_division_by_zero(self) -> None:
        params = [np.array([[1.0]]), np.array([[1.0]])]
        zero_grads = [np.array([[0.0]]), np.array([[0.0]])]

        adam = Adam(eps=1e-8)
        adam.step(params, zero_grads)

        assert not np.any(np.isnan(params[0]))
        assert not np.any(np.isnan(params[1]))
        assert not np.any(np.isinf(params[0]))
        assert not np.any(np.isinf(params[1]))

    def test_optimizer_gradient_descent_direction(self) -> None:
        params = [np.array([[1.0, -1.0]]), np.array([[0.5]])]
        positive_grads = [np.array([[0.1, 0.2]]), np.array([[0.1]])]
        negative_grads = [np.array([[-0.1, -0.2]]), np.array([[-0.1]])]

        basic = BasicOptimizer(learning_rate=0.1)

        params_pos = [p.copy() for p in params]
        params_neg = [p.copy() for p in params]

        basic.step(params_pos, positive_grads)
        basic.step(params_neg, negative_grads)

        assert np.all(params_pos[0] < params[0])
        assert np.all(params_neg[0] > params[0])
        assert params_pos[1] < params[1]
        assert params_neg[1] > params[1]
