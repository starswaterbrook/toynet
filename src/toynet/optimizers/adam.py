import numpy as np

from toynet.optimizers.protocol import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}

    def step(self, params: list[np.ndarray], grads: list) -> None:
        self.t += 1
        for p, g in zip(params, grads, strict=False):
            if g is None:
                continue

            key = id(p)

            m: np.ndarray = self.m.get(key, np.zeros_like(g))
            v: np.ndarray = self.v.get(key, np.zeros_like(g))

            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g**2)

            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            update = (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.eps)

            p[...] -= update

            self.m[key] = m
            self.v[key] = v
