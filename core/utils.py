# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import sys
import os


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


# Global logger instance
logger = logging.getLogger("ArcFoundry")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_sdk_version():
    """
    Simple parser to get version from pyproject.toml (SSOT).
    """
    try:
        # Locate project root (assuming core/utils.py is in core/)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        toml_path = os.path.join(root_dir, "pyproject.toml")

        if os.path.exists(toml_path):
            with open(toml_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("version"):
                        return line.split('=')[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "unknown"


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
    formatter = SmartNewlineFormatter(fmt=f"[ArcFoundry v{ver}] %(asctime)s [%(levelname)s] %(message)s",
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
