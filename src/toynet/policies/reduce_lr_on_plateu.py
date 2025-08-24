from toynet.networks.protocol import NeuralNetwork
from toynet.policies.protocol import Policy


class ReduceLROnPlateau(Policy):
    def __init__(self, factor: float = 0.1, patience: int = 10, min_lr: float = 1e-6) -> None:
        self.network: NeuralNetwork | None = None
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.waiting = 0
        self.best_validation_loss = float("inf")

    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network

    def __call__(self, validation_loss: float) -> bool:
        if not self.network:
            return False

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.waiting = 0
            return False
        self.waiting += 1
        if self.waiting >= self.patience:
            new_lr = self.network.optimizer.learning_rate * self.factor
            self.network.logger.info(
                "Reducing learning rate from %s to %s", self.network.optimizer.learning_rate, new_lr
            )
            if new_lr < self.min_lr:
                return True
            self.network.optimizer.learning_rate = new_lr
            self.waiting = 0
        return False

    def __str__(self) -> str:
        return str(self.__class__.__name__) + f"(min_lr={self.min_lr})"
