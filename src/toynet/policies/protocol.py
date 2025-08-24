from typing import Any, Protocol


class Policy(Protocol):
    network: Any | None

    def __call__(self, validation_loss: float) -> bool: ...

    def set_network(self, network: Any) -> None: ...
