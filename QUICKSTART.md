# Quick Start Guide

A step-by-step guide to get started with Face Mask Detection.

## 1. Setup (5 minutes)

```bash
# Clone and navigate
git clone https://github.com/aneessaheba/Face-Mask-Detection-Using-ML.git
cd Face-Mask-Detection-Using-ML

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data (10 minutes)

```bash
# Create data structure
mkdir -p data/mask data/without_mask

# Add your images:
# - Put masked face images in data/mask/
# - Put unmasked face images in data/without_mask/

# Recommended: Download from Kaggle
# https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
```

## 3. Train (varies based on data size)

```bash
# Using the Makefile (recommended)
make train DATA_DIR=data

# Or directly
python src/train.py --data-dir data --epochs 10 --batch-size 32
```

## 4. Evaluate

```bash
# Using the Makefile
make evaluate MODEL=models/mask_detector.h5 DATA_DIR=data

# Or directly
python src/evaluate.py --model models/mask_detector.h5 --data-dir data
```

## 5. Test on Your Images

```bash
# Single image
make infer MODEL=models/mask_detector.h5 IMAGE=your_image.jpg

# Live webcam
python src/live_infer.py --model models/mask_detector.h5 --camera 0
```

## Using Docker (Optional)

```bash
# Build image
make docker-build

# Run training in Docker
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  face-mask-detector train --data-dir /app/data
```

## Troubleshooting

### No images found in dataset
- Ensure folder structure is: `data/mask/*.jpg` and `data/without_mask/*.jpg`
- Check file extensions (must be .jpg, .jpeg, .png, .bmp, or .webp)

### Model evaluation shows low accuracy
- Ensure you have enough training data (ideally 1000+ images per class)
- Try increasing epochs: `--epochs 20` or `--epochs 30`
- Consider data augmentation or using a different model

### Webcam not working
- Check camera index (usually 0 for built-in camera)
- Ensure OpenCV can access your camera

## Next Steps

- Explore the [README.md](README.md) for complete documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
- Review [examples/example_workflow.py](examples/example_workflow.py) for an example pipeline

## Need Help?

- See the [README.md](README.md) for detailed documentation
- Check GitHub Issues for common questions
- Review the source code comments for implementation details

Happy detecting! 🎉
