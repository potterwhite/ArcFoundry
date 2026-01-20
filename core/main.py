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

import argparse
import sys
from core.utils import setup_logging, logger
from core.engine import PipelineEngine
import time


def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="ArcFoundry Core Engine V1.0")
    parser.add_argument('-c', '--config', required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Setup Logging
    # We can read a preliminary verbose flag from args if we wanted,
    # but for now we default to INFO and let Engine re-configure if needed.
    setup_logging(verbose=True)

    try:
        engine = PipelineEngine(args.config)
        engine.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred in the kernel: {e}")
        sys.exit(1)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    logger.info(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    main()
