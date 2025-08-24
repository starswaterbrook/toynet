import tempfile
from unittest.mock import Mock

import pytest

from toynet.policies.periodically_save_model import PeriodicallySaveModel
from toynet.policies.reduce_lr_on_plateu import ReduceLROnPlateau
from toynet.policies.save_best_model import SaveBestModel
from toynet.policies.validation_early_stop import ValidationLossEarlyStop


class TestPolicies:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        self.temp_file.close()
        self.save_path = self.temp_file.name

        self.mock_optimizer = Mock()
        self.mock_optimizer.learning_rate = 0.01

        self.mock_network = Mock()
        self.mock_network.save_to_npz = Mock()
        self.mock_network.optimizer = self.mock_optimizer

    def test_validation_early_stop_initialization(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=5, should_save=True)

        assert policy.save_path == self.save_path
        assert policy.patience == 5
        assert policy.should_save is True
        assert policy.best_loss == float("inf")
        assert policy.counter == 0
        assert policy.network is None

    def test_validation_early_stop_set_network(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=3)
        policy.set_network(self.mock_network)

        assert policy.network == self.mock_network

    def test_validation_early_stop_improving_loss(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=3)
        policy.set_network(self.mock_network)

        result1 = policy(1.0)
        assert result1 is False
        assert policy.best_loss == 1.0
        assert policy.counter == 0

        result2 = policy(0.8)
        assert result2 is False
        assert policy.best_loss == 0.8
        assert policy.counter == 0

    def test_validation_early_stop_patience_trigger(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=2, should_save=True)
        policy.set_network(self.mock_network)

        policy(1.0)
        assert policy.counter == 0

        policy(1.1)
        assert policy.counter == 1

        last_call = policy(1.2)
        assert policy.counter == 2
        assert last_call is True

        self.mock_network.save_to_npz.assert_called_with(self.save_path)

    def test_validation_early_stop_no_save(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=1, should_save=False)
        policy.set_network(self.mock_network)

        policy(1.0)
        result = policy(1.1)

        assert result is True
        self.mock_network.save_to_npz.assert_not_called()

    def test_validation_early_stop_reset_counter(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=3)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(1.1)
        policy(1.2)
        assert policy.counter == 2

        policy(0.9)
        assert policy.counter == 0
        assert policy.best_loss == 0.9

    def test_save_best_model_initialization(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=5)

        assert policy.save_path == self.save_path
        assert policy.grace_period == 5
        assert policy.best_validation_loss == float("inf")
        assert policy.waiting == 0
        assert policy.network is None

    def test_save_best_model_set_network(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=3)
        policy.set_network(self.mock_network)

        assert policy.network == self.mock_network

    def test_save_best_model_grace_period(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=3)
        policy.set_network(self.mock_network)

        result1 = policy(0.5)
        assert result1 is False
        assert policy.waiting == 1

        result2 = policy(0.4)
        assert result2 is False
        assert policy.waiting == 2

        result3 = policy(0.3)
        assert result3 is False
        assert policy.waiting == 3

        self.mock_network.save_to_npz.assert_not_called()

    def test_save_best_model_after_grace_period(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=2)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(0.9)

        result = policy(0.8)
        assert result is False
        assert policy.best_validation_loss == 0.8
        self.mock_network.save_to_npz.assert_called_with(self.save_path)

    def test_save_best_model_no_improvement(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=1)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(0.5)
        self.mock_network.save_to_npz.reset_mock()

        result = policy(0.6)
        assert result is False
        self.mock_network.save_to_npz.assert_not_called()

    def test_reduce_lr_on_plateau_initialization(self) -> None:
        policy = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)

        assert policy.factor == 0.5
        assert policy.patience == 3
        assert policy.min_lr == 1e-5
        assert policy.waiting == 0
        assert policy.best_validation_loss == float("inf")
        assert policy.network is None

    def test_reduce_lr_on_plateau_set_network(self) -> None:
        policy = ReduceLROnPlateau()
        policy.set_network(self.mock_network)

        assert policy.network == self.mock_network

    def test_reduce_lr_on_plateau_improving_loss(self) -> None:
        policy = ReduceLROnPlateau(patience=2)
        policy.set_network(self.mock_network)

        result1 = policy(1.0)
        assert result1 is False
        assert policy.best_validation_loss == 1.0
        assert policy.waiting == 0

        result2 = policy(0.8)
        assert result2 is False
        assert policy.best_validation_loss == 0.8
        assert policy.waiting == 0

    def test_reduce_lr_on_plateau_patience_trigger(self) -> None:
        policy = ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-8)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(1.1)
        assert policy.waiting == 1

        result = policy(1.2)
        assert result is False
        assert policy.waiting == 0
        assert self.mock_network.optimizer.learning_rate == 0.005

    def test_reduce_lr_on_plateau_min_lr_limit(self) -> None:
        self.mock_network.optimizer.learning_rate = 1e-5
        policy = ReduceLROnPlateau(factor=0.1, patience=1, min_lr=1e-5)
        policy.set_network(self.mock_network)

        policy(1.0)
        result = policy(1.1)

        assert result is True
        assert self.mock_network.optimizer.learning_rate == 1e-5

    def test_periodically_save_model_initialization(self) -> None:
        policy = PeriodicallySaveModel(self.save_path, save_interval=5)

        assert policy.save_path == self.save_path
        assert policy.save_interval == 5
        assert policy.current_step == 0
        assert policy.network is None

    def test_periodically_save_model_set_network(self) -> None:
        policy = PeriodicallySaveModel(self.save_path, save_interval=3)
        policy.set_network(self.mock_network)

        assert policy.network == self.mock_network

    def test_periodically_save_model_save_interval(self) -> None:
        policy = PeriodicallySaveModel(self.save_path, save_interval=3)
        policy.set_network(self.mock_network)

        result1 = policy(1.0)
        assert result1 is False
        assert policy.current_step == 1
        self.mock_network.save_to_npz.assert_called_with(self.save_path)

        self.mock_network.save_to_npz.reset_mock()

        result2 = policy(0.9)
        assert result2 is False
        assert policy.current_step == 2
        self.mock_network.save_to_npz.assert_not_called()

        result3 = policy(0.8)
        assert result3 is False
        assert policy.current_step == 3
        self.mock_network.save_to_npz.assert_not_called()

        result4 = policy(1.0)
        assert result4 is False
        assert policy.current_step == 4
        self.mock_network.save_to_npz.assert_called_with(self.save_path)

    def test_periodically_save_model_ignores_validation_loss(self) -> None:
        policy = PeriodicallySaveModel(self.save_path, save_interval=2)
        policy.set_network(self.mock_network)

        policy(float("inf"))
        policy(float("-inf"))

        assert self.mock_network.save_to_npz.call_count == 1

    def test_policy_interface_consistency(self) -> None:
        policies = [
            ValidationLossEarlyStop(self.save_path, patience=3),
            SaveBestModel(self.save_path, save_grace_period=2),
            ReduceLROnPlateau(),
            PeriodicallySaveModel(self.save_path, save_interval=5),
        ]

        for policy in policies:
            assert hasattr(policy, "network")
            assert hasattr(policy, "set_network")
            assert callable(policy.set_network)
            assert callable(policy)

    def test_policy_string_representations(self) -> None:
        early_stop = ValidationLossEarlyStop(self.save_path, patience=5)
        save_best = SaveBestModel(self.save_path, save_grace_period=3)
        reduce_lr = ReduceLROnPlateau(min_lr=1e-6)
        periodic_save = PeriodicallySaveModel(self.save_path, save_interval=10)

        assert "ValidationLossEarlyStop" in str(early_stop)
        assert "patience=5" in str(early_stop)
        assert "SaveBestModel" in str(save_best)
        assert "ReduceLROnPlateau" in str(reduce_lr)
        assert "min_lr=1e-06" in str(reduce_lr)
        assert "PeriodicallySaveModel" in str(periodic_save)

    def test_save_best_model_zero_grace_period(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=0)
        policy.set_network(self.mock_network)

        result = policy(1.0)
        assert result is False
        self.mock_network.save_to_npz.assert_called_with(self.save_path)

    def test_periodically_save_model_interval_one(self) -> None:
        policy = PeriodicallySaveModel(self.save_path, save_interval=1)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(2.0)
        policy(3.0)

        assert self.mock_network.save_to_npz.call_count == 3

    def test_multiple_policy_coordination(self) -> None:
        early_stop = ValidationLossEarlyStop(self.save_path, patience=2)
        save_best = SaveBestModel(self.save_path, save_grace_period=1)

        early_stop.set_network(self.mock_network)
        save_best.set_network(self.mock_network)

        losses = [1.0, 1.1, 1.2]
        early_stop_triggered = False

        for loss in losses:
            if early_stop(loss):
                early_stop_triggered = True
                break
            save_best(loss)

        assert early_stop_triggered is True
        assert self.mock_network.save_to_npz.call_count == 2

    def test_reduce_lr_multiple_reductions(self) -> None:
        policy = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6)
        policy.set_network(self.mock_network)

        initial_lr = self.mock_network.optimizer.learning_rate

        policy(1.0)
        policy(1.1)
        assert self.mock_network.optimizer.learning_rate == initial_lr * 0.5

        policy(1.2)
        assert self.mock_network.optimizer.learning_rate == initial_lr * 0.25

    def test_save_best_model_multiple_improvements(self) -> None:
        policy = SaveBestModel(self.save_path, save_grace_period=1)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(0.8)
        policy(0.6)
        policy(0.4)

        assert self.mock_network.save_to_npz.call_count == 3

    def test_early_stop_recovery_behavior(self) -> None:
        policy = ValidationLossEarlyStop(self.save_path, patience=3)
        policy.set_network(self.mock_network)

        policy(1.0)
        policy(1.1)
        policy(1.2)
        assert policy.counter == 2

        policy(0.9)
        assert policy.counter == 0

        policy(1.0)
        policy(1.1)
        assert policy.counter == 2

        result = policy(1.2)
        assert result is True

    def test_policy_return_values_consistency(self) -> None:
        policies = [
            ValidationLossEarlyStop(self.save_path, patience=10),
            SaveBestModel(self.save_path, save_grace_period=1),
            ReduceLROnPlateau(patience=10),
            PeriodicallySaveModel(self.save_path, save_interval=5),
        ]

        for policy in policies:
            policy.set_network(self.mock_network)
            result = policy(1.0)
            assert isinstance(result, bool)
