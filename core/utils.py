import logging
import sys
import os

def setup_logging(verbose=False):
    """Configures the global logger."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create a custom formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("ArcFoundry")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger

# Global logger instance
logger = logging.getLogger("ArcFoundry")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
