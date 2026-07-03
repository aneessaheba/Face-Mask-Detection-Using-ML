import os
import logging.config
import json


def setup_logging(config_path: str = "config/logging.json", log_level: str = "INFO"):
    """Initialize logging from config or defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            logging.config.dictConfig(json.load(f))
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance by name."""
    return logging.getLogger(name)
