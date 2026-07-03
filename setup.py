from setuptools import setup, find_packages

setup(
    name="face-mask-detector",
    version="0.1.0",
    description="Face mask detection using convolutional neural networks",
    author="aneessaheba",
    author_email="aneessaheba.guddi@sjsu.edu",
    url="https://github.com/aneessaheba/Face-Mask-Detection-Using-ML",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "tensorflow>=2.14.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mask-detector-train=src.cli:train",
            "mask-detector-eval=src.cli:evaluate",
            "mask-detector-infer=src.cli:infer",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
