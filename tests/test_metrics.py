import json
import os
import tempfile
import unittest

import numpy as np

from src.metrics import compute_binary_metrics, save_metrics_report


class MetricsTests(unittest.TestCase):
    def test_compute_binary_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.2, 0.8, 0.3])

        metrics = compute_binary_metrics(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_save_metrics_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metrics.json")
            metrics = {"accuracy": 0.95, "precision": 0.9}
            saved_path = save_metrics_report(metrics, output_path)

            self.assertEqual(saved_path, output_path)
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
