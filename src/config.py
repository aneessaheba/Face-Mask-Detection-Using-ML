import os
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    """Application configuration."""
    data_dir: str = "data"
    model_dir: str = "models"
    report_dir: str = "reports"
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 10
    test_split: float = 0.2
    threshold: float = 0.5
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def ensure_dirs(self):
        """Create necessary directories."""
        for attr in ["data_dir", "model_dir", "report_dir"]:
            path = getattr(self, attr)
            os.makedirs(path, exist_ok=True)


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()
