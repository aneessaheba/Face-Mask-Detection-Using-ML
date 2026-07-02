# Face Mask Detection Using ML

A complete starter project for building and evaluating a face-mask detection system with Python, OpenCV, TensorFlow/Keras, and basic experiment metrics.

## What this project includes

- A lightweight CNN model for binary classification: mask vs. no mask
- Image loading and preprocessing utilities
- Training and evaluation scripts
- Image inference and webcam-based live inference
- Metric computation and JSON report export for accuracy, precision, recall, F1 score, and confusion matrix
- A small test suite for the metrics layer

## Project structure

- [src/data_utils.py](src/data_utils.py) – dataset loading and image preprocessing helpers
- [src/model.py](src/model.py) – convolutional neural network definition
- [src/train.py](src/train.py) – training script with history export
- [src/evaluate.py](src/evaluate.py) – evaluation script with metrics export
- [src/infer.py](src/infer.py) – single-image inference
- [src/live_infer.py](src/live_infer.py) – webcam-based live inference
- [src/metrics.py](src/metrics.py) – binary classification metrics and JSON report writing
- [tests/test_metrics.py](tests/test_metrics.py) – unit tests for the metrics layer

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

Organize your dataset like this:

```text
data/
  mask/
    *.jpg
  without_mask/
    *.jpg
```

The loader expects folder names that include `mask` or `without_mask` and will label them accordingly.

## Training

```bash
python src/train.py --data-dir data --epochs 10 --batch-size 32 --output models/mask_detector.h5 --history-output reports/training_history.json
```

This will:
- train the CNN model,
- save the best checkpoint to `models/mask_detector.h5`,
- export training history to `reports/training_history.json`.

## Evaluation

```bash
python src/evaluate.py --model models/mask_detector.h5 --data-dir data --metrics-output reports/evaluation_metrics.json
```

This exports evaluation metrics such as:
- accuracy
- precision
- recall
- F1 score
- confusion matrix
- test loss and test accuracy

## Inference

Single image:

```bash
python src/infer.py --model models/mask_detector.h5 --image sample.jpg --output output.jpg
```

Live webcam inference:

```bash
python src/live_infer.py --model models/mask_detector.h5 --camera 0
```

## Testing

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- This is a strong starter template for a computer-vision project and can be extended with better preprocessing, data augmentation, transfer learning, and deployment workflows.
- For best results, use a larger real-world dataset such as a Kaggle face mask detection dataset or a custom collection of labeled images.
