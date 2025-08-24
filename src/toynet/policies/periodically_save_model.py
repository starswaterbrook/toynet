from toynet.networks.protocol import NeuralNetwork
from toynet.policies.protocol import Policy


class PeriodicallySaveModel(Policy):
    def __init__(self, save_path: str, save_interval: int) -> None:
        self.network: NeuralNetwork | None = None
        self.save_path = save_path
        self.save_interval = save_interval
        self.current_step = 0

    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network

    def __call__(self, _: float) -> bool:
        if self.current_step % self.save_interval == 0 and self.network:
            self.network.logger.info("Saving model to <%s>", self.save_path)
            self.network.save_to_npz(self.save_path)
        self.current_step += 1
        return False

    def __str__(self) -> str:
        return str(self.__class__.__name__)
