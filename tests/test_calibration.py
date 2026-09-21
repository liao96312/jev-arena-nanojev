import math
import unittest

from scripts.calibrate_predictions import fit_temperature, metrics, probabilities


class CalibrationTests(unittest.TestCase):
    def test_fits_known_soft_target_temperature(self):
        logits = [2.0, 0.0, -1.0]
        target = probabilities(logits, 2.0)
        rows = [{"student_logits": logits, "gold_distribution_probs": target} for _ in range(4)]
        temperature, fitted = fit_temperature(rows)
        self.assertTrue(fitted)
        self.assertAlmostEqual(temperature, 2.0, places=5)
        self.assertLess(metrics(rows, temperature)["target_ce"], metrics(rows, 1.0)["target_ce"])
