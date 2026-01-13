import argparse
import sys
from core.utils import setup_logging, logger
from core.engine import PipelineEngine

def main():
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

if __name__ == "__main__":
    main()
