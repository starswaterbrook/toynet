import logging

import numpy as np

from toynet.config import get_simple_logger
from toynet.data_loaders import DataLoader
from toynet.functions import LossFunction
from toynet.layers import Layer
from toynet.optimizers import Optimizer
from toynet.policies.protocol import Policy


class MultiLayerPerceptron:
    def __init__(
        self,
        layers: list[Layer],
        loss_function: type[LossFunction],
        optimizer: Optimizer,
        logger: logging.Logger | None = None,
    ) -> None:
        self._layers: list[Layer] = layers
        self.loss_function: LossFunction = loss_function()
        self.optimizer: Optimizer = optimizer

        if logger is None:
            self.logger = get_simple_logger(__name__)
        else:
            self.logger = logger

    def forward(self, network_input: np.ndarray) -> np.ndarray:
        curr_vector = network_input.copy()
        for layer in self._layers:
            curr_vector = layer.forward(curr_vector)
        return curr_vector

    def backprop(self, network_input: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = self.forward(network_input)

        grad = self.loss_function(y_true, y_pred, derivative=True)
        for layer in reversed(self._layers):
            grad = layer.backpass(grad)
        return grad

    def train(  # noqa: C901
        self, data_loader: DataLoader, epochs: int, policies: list[Policy] | None = None
    ) -> None:
        if policies is None:
            policies = []
        for policy in policies:
            policy.set_network(self)

        self.logger.info("Starting training...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in data_loader.train_generator():
                for layer in self._layers:
                    layer.reset_accumulated_gradient()

                batch_loss = 0.0

                for i in range(len(batch_x)):
                    out = self.forward(batch_x[i])
                    batch_loss += float(self.loss_function(batch_y[i], out)[0][0])
                    self.backprop(batch_x[i], batch_y[i])

                for layer in self._layers:
                    averaged_grads = [grad / len(batch_x) for grad in layer.gradients]
                    self.optimizer.step(layer.parameters, averaged_grads)

                total_loss += batch_loss
            self.logger.info(
                "Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, total_loss / data_loader.n_samples
            )
            if data_loader.n_val > 0:
                val_loss = 0.0
                for X_val, y_val in data_loader.val_generator():
                    for i in range(len(X_val)):
                        out = self.forward(X_val[i])
                        val_loss += float(self.loss_function(y_val[i], out)[0][0])
                avg_val_loss = val_loss / data_loader.n_val
                for policy in policies:
                    if policy(avg_val_loss):
                        self.logger.info(
                            "Stopping training at epoch %d due to policy %s", epoch + 1, policy
                        )
                        return
                self.logger.info(
                    "Epoch %d/%d - Validation Loss: %.4f",
                    epoch + 1,
                    epochs,
                    avg_val_loss,
                )

    def save_to_npz(self, filename: str) -> None:
        data = {}
        for i, layer in enumerate(self._layers):
            weights, bias = layer.parameters
            data[f"W{i}"] = weights
            data[f"b{i}"] = bias
        np.savez(filename, **data)  # type: ignore[arg-type]

    def load_from_npz(self, filename: str) -> None:
        archive = np.load(filename, allow_pickle=True)
        for i, layer in enumerate(self._layers):
            layer._weights = archive[f"W{i}"]  # type: ignore[attr-defined] # noqa: SLF001
            layer._bias = archive[f"b{i}"]  # type: ignore[attr-defined] # noqa: SLF001

    def __call__(self, network_input: np.ndarray) -> np.ndarray:
        return self.forward(network_input)
