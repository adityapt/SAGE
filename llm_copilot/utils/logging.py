"""Logging configuration"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "llm_copilot",
    level: int = logging.INFO,
    log_file: str = None
) -> logging.Logger:
    """
    Configure logger for the application.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level
    log_file : str, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

