# Face Mask Detection Using ML

This project provides a starter pipeline for building a face mask detection system with Python, OpenCV, and TensorFlow/Keras.

## Project structure

- `src/data_utils.py` – dataset loading and preprocessing helpers
- `src/model.py` – CNN model definition
- `src/train.py` – training script
- `src/infer.py` – image inference script with face detection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Organize your dataset into two folders:

```text
data/
  mask/
    *.jpg
  without_mask/
    *.jpg
```

Then run:

```bash
python src/train.py --data-dir data --epochs 10 --batch-size 32 --output models/mask_detector.h5
```

## Inference

```bash
python src/infer.py --model models/mask_detector.h5 --image sample.jpg --output output.jpg
```

## Notes

- The project is intentionally structured as a clean starter template.
- For best results, use a real face-mask image dataset such as the Kaggle Face Mask Detection dataset.
