import numpy as np
import pytest

from toynet.initalizers.he import He
from toynet.initalizers.xavier import Xavier


class TestInitializers:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        np.random.seed(1)

        self.small_in = 2
        self.small_out = 3
        self.medium_in = 10
        self.medium_out = 5
        self.large_in = 100
        self.large_out = 50
        self.very_large_in = 10000
        self.very_large_out = 5000

        self.test_dimensions = [
            (1, 1),
            (2, 3),
            (10, 5),
            (100, 50),
            (784, 128),
            (128, 10),
        ]

        self.he = He()
        self.xavier = Xavier()

    def test_he_weight_initialization_shape(self) -> None:
        for in_amount, out_amount in self.test_dimensions:
            weights = self.he.initialize_weights(in_amount, out_amount)
            assert weights.shape == (in_amount, out_amount)

    def test_he_bias_initialization_shape(self) -> None:
        for _, out_amount in self.test_dimensions:
            bias = self.he.initialize_bias(out_amount)
            assert bias.shape == (1, out_amount)

    def test_xavier_weight_initialization_shape(self) -> None:
        for in_amount, out_amount in self.test_dimensions:
            weights = self.xavier.initialize_weights(in_amount, out_amount)
            assert weights.shape == (in_amount, out_amount)

    def test_xavier_bias_initialization_shape(self) -> None:
        for _, out_amount in self.test_dimensions:
            bias = self.xavier.initialize_bias(out_amount)
            assert bias.shape == (1, out_amount)

    def test_he_weight_distribution_properties(self) -> None:
        weights = self.he.initialize_weights(1000, 100)

        expected_std = np.sqrt(2.0 / 1000)
        actual_std = np.std(weights)

        assert abs(actual_std - expected_std) < 0.01
        assert abs(np.mean(weights)) < 0.01

    def test_xavier_weight_distribution_properties(self) -> None:
        weights = self.xavier.initialize_weights(1000, 100)

        expected_std = np.sqrt(2.0 / (1000 + 100))
        actual_std = np.std(weights)

        assert abs(actual_std - expected_std) < 0.01
        assert abs(np.mean(weights)) < 0.01

    def test_he_vs_xavier_different_scaling(self) -> None:
        he_weights = self.he.initialize_weights(100, 50)
        xavier_weights = self.xavier.initialize_weights(100, 50)

        he_std = np.std(he_weights)
        xavier_std = np.std(xavier_weights)

        assert he_std > xavier_std

    def test_initializer_call_method(self) -> None:
        for in_amount, out_amount in self.test_dimensions:
            he_weights, he_bias = self.he(in_amount, out_amount)
            xavier_weights, xavier_bias = self.xavier(in_amount, out_amount)

            assert he_weights.shape == (in_amount, out_amount)
            assert he_bias.shape == (1, out_amount)
            assert xavier_weights.shape == (in_amount, out_amount)
            assert xavier_bias.shape == (1, out_amount)

    def test_initializer_reproducibility(self) -> None:
        np.random.seed(123)
        he_weights_1 = self.he.initialize_weights(10, 5)

        np.random.seed(123)
        he_weights_2 = self.he.initialize_weights(10, 5)

        np.testing.assert_array_equal(he_weights_1, he_weights_2)

        np.random.seed(456)
        xavier_weights_1 = self.xavier.initialize_weights(10, 5)

        np.random.seed(456)
        xavier_weights_2 = self.xavier.initialize_weights(10, 5)

        np.testing.assert_array_equal(xavier_weights_1, xavier_weights_2)

    def test_initializer_variance_scaling_correctness(self) -> None:
        test_cases = [(50, 25), (100, 50), (200, 100)]

        for in_amount, out_amount in test_cases:
            he_weights = self.he.initialize_weights(in_amount, out_amount)
            he_variance = np.var(he_weights)
            expected_he_variance = 2.0 / in_amount

            xavier_weights = self.xavier.initialize_weights(in_amount, out_amount)
            xavier_variance = np.var(xavier_weights)
            expected_xavier_variance = 2.0 / (in_amount + out_amount)

            assert abs(he_variance - expected_he_variance) < 0.05
            assert abs(xavier_variance - expected_xavier_variance) < 0.05

    def test_weight_initialization_non_zero(self) -> None:
        for in_amount, out_amount in self.test_dimensions[:3]:
            he_weights = self.he.initialize_weights(in_amount, out_amount)
            xavier_weights = self.xavier.initialize_weights(in_amount, out_amount)

            assert not np.all(he_weights == 0)
            assert not np.all(xavier_weights == 0)

    def test_different_initializations_are_different(self) -> None:
        he_weights_1 = self.he.initialize_weights(10, 5)
        he_weights_2 = self.he.initialize_weights(10, 5)

        assert not np.array_equal(he_weights_1, he_weights_2)

        xavier_weights_1 = self.xavier.initialize_weights(10, 5)
        xavier_weights_2 = self.xavier.initialize_weights(10, 5)

        assert not np.array_equal(xavier_weights_1, xavier_weights_2)

    def test_he_xavier_produce_different_weights(self) -> None:
        he_weights = self.he.initialize_weights(10, 5)
        xavier_weights = self.xavier.initialize_weights(10, 5)

        assert not np.array_equal(he_weights, xavier_weights)

    def test_edge_case_single_neuron(self) -> None:
        he_weights = self.he.initialize_weights(1, 1)
        xavier_weights = self.xavier.initialize_weights(1, 1)

        assert he_weights.shape == (1, 1)
        assert xavier_weights.shape == (1, 1)
        assert not np.isnan(he_weights).any()
        assert not np.isnan(xavier_weights).any()

    def test_weight_distribution_normality(self) -> None:
        large_sample_he = self.he.initialize_weights(1000, 1000)
        large_sample_xavier = self.xavier.initialize_weights(1000, 1000)

        he_skewness = abs(
            np.mean(((large_sample_he - np.mean(large_sample_he)) / np.std(large_sample_he)) ** 3)
        )
        xavier_skewness = abs(
            np.mean(
                ((large_sample_xavier - np.mean(large_sample_xavier)) / np.std(large_sample_xavier))
                ** 3
            )
        )

        assert he_skewness < 0.2
        assert xavier_skewness < 0.2

    def test_initializer_interface_consistency(self) -> None:
        initializers = [self.he, self.xavier]

        for init in initializers:
            weights = init.initialize_weights(5, 3)
            bias = init.initialize_bias(3)
            weights_bias_tuple = init(5, 3)

            assert isinstance(weights, np.ndarray)
            assert isinstance(bias, np.ndarray)
            assert isinstance(weights_bias_tuple, tuple)
            assert len(weights_bias_tuple) == 2

    def test_weight_initialization_statistical_properties(self) -> None:
        sample_size = 10000

        he_large_sample = self.he.initialize_weights(sample_size, 1).flatten()
        xavier_large_sample = self.xavier.initialize_weights(sample_size, 1).flatten()

        assert abs(np.mean(he_large_sample)) < 0.01
        assert abs(np.mean(xavier_large_sample)) < 0.01

        he_expected_std = np.sqrt(2.0 / sample_size)
        xavier_expected_std = np.sqrt(2.0 / (sample_size + 1))

        assert abs(np.std(he_large_sample) - he_expected_std) < 0.001
        assert abs(np.std(xavier_large_sample) - xavier_expected_std) < 0.001

    def test_large_dimension_stability(self) -> None:
        he_weights = self.he.initialize_weights(self.very_large_in, self.very_large_out)
        xavier_weights = self.xavier.initialize_weights(self.very_large_in, self.very_large_out)

        assert not np.any(np.isnan(he_weights))
        assert not np.any(np.isinf(he_weights))
        assert not np.any(np.isnan(xavier_weights))
        assert not np.any(np.isinf(xavier_weights))

        assert he_weights.shape == (self.very_large_in, self.very_large_out)
        assert xavier_weights.shape == (self.very_large_in, self.very_large_out)
