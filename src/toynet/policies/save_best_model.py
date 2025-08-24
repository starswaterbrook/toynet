from toynet.networks.protocol import NeuralNetwork
from toynet.policies.protocol import Policy


class SaveBestModel(Policy):
    def __init__(self, save_path: str, save_grace_period: int) -> None:
        self.network: NeuralNetwork | None = None
        self.save_path = save_path
        self.best_validation_loss = float("inf")
        self.grace_period = save_grace_period
        self.waiting = 0

    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network

    def __call__(self, validation_loss: float) -> bool:
        if self.waiting < self.grace_period:
            self.waiting += 1
            return False

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            if self.network:
                self.network.save_to_npz(self.save_path)
        return False

    def __str__(self) -> str:
        return str(self.__class__.__name__)
