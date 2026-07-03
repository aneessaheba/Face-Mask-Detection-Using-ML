import os
import sys
from src.logger import setup_logging, get_logger
from src.config import get_default_config

logger = get_logger(__name__)


def main():
    """Main entry point for the Face Mask Detection application."""
    setup_logging()
    
    config = get_default_config()
    config.ensure_dirs()
    
    logger.info("Face Mask Detection Application Started")
    logger.info(f"Configuration: data_dir={config.data_dir}, model_dir={config.model_dir}")
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   Face Mask Detection Using ML             ║
    ║   Production-Ready ML Application          ║
    ╚════════════════════════════════════════════╝
    
    Commands:
        python -m src.cli train     - Train the model
        python -m src.cli evaluate  - Evaluate the model
        python -m src.cli infer     - Run inference on an image
        python src/live_infer.py    - Run live webcam detection
    
    For more information, see README.md
    """)
    
    logger.info("Application ready. Run 'python -m src.cli --help' for available commands.")


if __name__ == "__main__":
    main()
