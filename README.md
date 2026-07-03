# Face Mask Detection Using ML

[![tests](https://github.com/aneessaheba/Face-Mask-Detection-Using-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/aneessaheba/Face-Mask-Detection-Using-ML/actions/workflows/tests.yml)

A production-ready starter project for building and deploying a face-mask detection system using convolutional neural networks. This project includes everything needed for model training, evaluation, inference, and deployment.

## 🎯 Features

- **CNN Model**: Lightweight convolutional neural network for binary classification (mask vs. no mask)
- **Training Pipeline**: End-to-end training with history export and best model checkpointing
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Inference**: Single-image and real-time webcam-based detection
- **Logging & Configuration**: Production-ready logging and configuration management
- **Testing**: Unit tests for core functionality
- **Containerization**: Docker and Docker Compose support for easy deployment
- **CI/CD**: GitHub Actions workflow for automated testing
- **CLI Interface**: Command-line tools for all major operations
- **Documentation**: Complete guides for setup, usage, and contribution

## 📁 Project Structure

```
Face-Mask-Detection-Using-ML/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   ├── logger.py                # Logging setup
│   ├── data_utils.py            # Dataset loading and preprocessing
│   ├── model.py                 # CNN model definition
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── infer.py                 # Single-image inference
│   ├── live_infer.py            # Webcam-based inference
│   └── metrics.py               # Metrics computation and reporting
├── tests/                        # Unit tests
│   └── test_metrics.py
├── config/                       # Configuration files
│   ├── default.json
│   └── logging.json
├── .github/workflows/            # GitHub Actions workflows
│   └── tests.yml
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── setup.py                      # Package setup
├── pyproject.toml               # Project metadata and tools config
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── .env.example                 # Environment variables template
├── .gitignore
├── README.md                    # This file
├── LICENSE                      # MIT License
└── CONTRIBUTING.md              # Contribution guidelines
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aneessaheba/Face-Mask-Detection-Using-ML.git
cd Face-Mask-Detection-Using-ML
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

Organize your dataset as follows:

```
data/
├── mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Recommended datasets:
- [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [Simulated Masked Face Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

## 🎓 Usage

### Training

Using the CLI:
```bash
python -m src.cli train --data-dir data --epochs 10 --batch-size 32 --output models/mask_detector.h5
```

Or directly:
```bash
python src/train.py --data-dir data --epochs 10 --batch-size 32 --output models/mask_detector.h5
```

Training history will be saved to `reports/training_history.json`.

### Evaluation

```bash
python -m src.cli evaluate --model models/mask_detector.h5 --data-dir data
```

Evaluation metrics will be saved to `reports/evaluation_metrics.json`.

### Single Image Inference

```bash
python -m src.cli infer --model models/mask_detector.h5 --image sample.jpg --output output.jpg
```

### Live Webcam Inference

```bash
python src/live_infer.py --model models/mask_detector.h5 --camera 0
```

Press `q` to quit.

## 🧪 Testing

Run all tests:
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run with verbose output:
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## 🐳 Docker

### Build and Run

Build the Docker image:
```bash
docker build -t face-mask-detector .
```

Run a container:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models face-mask-detector train --data-dir /app/data --output /app/models/mask_detector.h5
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 📦 Installation as a Package

Install the package in development mode:
```bash
pip install -e .
```

Or with development dependencies:
```bash
pip install -e ".[dev]"
```

Then use the command-line tools:
```bash
mask-detector-train --data-dir data --epochs 10
mask-detector-eval --model models/mask_detector.h5 --data-dir data
mask-detector-infer --model models/mask_detector.h5 --image sample.jpg
```

## ⚙️ Configuration

Configuration is managed through:
1. `config/default.json` - Default settings
2. `.env` file - Environment-specific overrides
3. Command-line arguments - Runtime overrides

Copy `.env.example` to `.env` and adjust as needed.

## 📝 Metrics

Evaluation metrics include:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN breakdown

## 🔄 CI/CD

The project includes automated testing via GitHub Actions. Tests run on:
- Python 3.8, 3.9, 3.10
- All push events and pull requests to main/develop branches

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Code style and testing requirements

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**aneessaheba**
- GitHub: [@aneessaheba](https://github.com/aneessaheba)
- Email: aneessaheba.guddi@sjsu.edu

## 🙏 Acknowledgments

- TensorFlow/Keras documentation and examples
- OpenCV community
- Kaggle datasets community

## 📚 Further Reading

- [TensorFlow CNN Guide](https://www.tensorflow.org/tutorials/images/cnn)
- [Face Detection with OpenCV](https://docs.opencv.org/master/d3/dc1/tutorial_simple_linear_classifier.html)
- [Binary Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Happy coding!** 🎉
