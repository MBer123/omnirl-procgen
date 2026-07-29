import logging


def setup_logger(name="OmniRL"):
    """Create and configure a logger with a stream handler."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(levelname)s:%(process)d %(asctime)s] %(message)s")
    )
    logger.addHandler(h)
    return logger
