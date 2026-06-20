import logging

LOGGER_NAME = "adult_income_api"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level)
    return logging.getLogger(LOGGER_NAME)


logger = logging.getLogger(LOGGER_NAME)
