"""Rich terminal logging for training and model evaluation."""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def get_logger(name: str = "ParagAI") -> logging.Logger:
    """Return a standard logger formatted with Rich."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

