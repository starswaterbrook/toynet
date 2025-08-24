import logging

NUMERICAL_EPS = 1e-12


def get_simple_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] - %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)

    logger.propagate = False
    return logger
