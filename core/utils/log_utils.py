import logging
import sys
from .file_utils import get_sdk_version

# Global logger instance
logger = logging.getLogger("ArcFoundry")


# 1. Define the Custom Formatter
class SmartNewlineFormatter(logging.Formatter):
    """
    A custom formatter that detects leading newlines in the log message
    and moves them to the very beginning of the final output string,
    before the log prefix (timestamp, level, etc.).
    """

    def format(self, record):
        # Ensure message is a string
        original_msg = str(record.msg)

        # Check for leading newlines
        if original_msg.startswith('\n'):
            # Calculate the number of leading newlines
            stripped_msg = original_msg.lstrip('\n')
            newline_count = len(original_msg) - len(stripped_msg)
            prefix_newlines = '\n' * newline_count

            # --- The Trick ---
            # 1. Temporarily strip newlines from the record message
            record.msg = stripped_msg

            # 2. Let the parent class format the standard line (Prefix + Message)
            formatted_line = super().format(record)

            # 3. Restore the original message (good practice for other handlers)
            record.msg = original_msg

            # 4. Prepend the newlines to the very front
            return prefix_newlines + formatted_line

        # Default behavior for messages without leading newlines
        return super().format(record)


def setup_logging(verbose=False):
    """
    Configure logging with version info and verbosity control.
    """
    ver = get_sdk_version()

    # Determine Log Level
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create Formatter
    # formatter = logging.Formatter(
    #     fmt=f"[ArcFoundry v{ver}] %(asctime)s [%(levelname)s] %(message)s",
    #     datefmt="%H:%M:%S"
    # )
    formatter = SmartNewlineFormatter(
        fmt=f"[ArcFoundry v{ver}] %(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S")

    # Setup Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if re-initialized
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    return logging.getLogger("ArcFoundry")


# Initialize a default logger instance for module-level usage
logger = setup_logging(verbose=False)
