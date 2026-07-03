"""
Example script demonstrating the Face Mask Detection pipeline.
This script shows a complete workflow from data preparation to evaluation.
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logger import setup_logging, get_logger
from src.data_utils import load_dataset, split_dataset
from src.model import build_model
from src.metrics import save_metrics_report

logger = get_logger(__name__)


def main():
    """Run a complete example workflow."""
    setup_logging()
    
    # Configuration
    config = Config(
        data_dir="data",
        model_dir="models",
        report_dir="reports",
        epochs=2,  # Small number for demo
        batch_size=16,
    )
    config.ensure_dirs()
    
    logger.info("Starting Face Mask Detection Example Workflow")
    logger.info(f"Configuration: {config}")
    
    # Step 1: Load and prepare data
    logger.info("Loading dataset...")
    try:
        images, labels = load_dataset(config.data_dir, image_size=(config.image_size, config.image_size))
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Please ensure your dataset is in the correct format:")
        logger.info("  data/mask/*.jpg")
        logger.info("  data/without_mask/*.jpg")
        return
    
    # Step 2: Split data
    logger.info("Splitting dataset...")
    x_train, x_val, y_train, y_val = split_dataset(images, labels, test_size=config.test_split)
    logger.info(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    # Step 3: Build and train model
    logger.info("Building model...")
    model = build_model(input_shape=(config.image_size, config.image_size, 3))
    
    logger.info("Training model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
    )
    
    # Step 4: Save results
    logger.info("Saving model and metrics...")
    model_path = os.path.join(config.model_dir, "demo_model.h5")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(config.report_dir, "demo_history.json")
    save_metrics_report(history.history, history_path)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("Example workflow completed successfully!")


if __name__ == "__main__":
    main()
