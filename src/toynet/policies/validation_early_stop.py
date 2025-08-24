from toynet.networks.protocol import NeuralNetwork
from toynet.policies.protocol import Policy


class ValidationLossEarlyStop(Policy):
    def __init__(self, save_path: str, patience: int, should_save: bool = True) -> None:
        self.network: NeuralNetwork | None = None
        self.save_path = save_path
        self.patience = patience
        self.best_loss = float("inf")
        self.should_save = should_save
        self.counter = 0

    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network

    def _should_stop(self, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def __call__(self, validation_loss: float) -> bool:
        if self._should_stop(validation_loss):
            if self.should_save and self.network:
                self.network.save_to_npz(self.save_path)
            return True
        return False

    def __str__(self) -> str:
        return str(self.__class__.__name__) + f"(patience={self.patience})"
