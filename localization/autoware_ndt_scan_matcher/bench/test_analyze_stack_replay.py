#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).with_name("analyze_stack_replay.py")
SPEC = importlib.util.spec_from_file_location("analyze_stack_replay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pose(x, y=0.0, yaw=0.0):
    return {"x": str(x), "y": str(y), "yaw": str(yaw)}


def frame(sequence, raw_x, ekf_x, quality=True):
    return {
        "sequence": sequence,
        "stamp_ns": sequence * 100_000_000,
        "raw": pose(raw_x),
        "diag": {},
        "quality_pass": quality,
        "accepted": pose(raw_x) if quality else None,
        "ekf_update": None,
        "ekf_output": pose(ekf_x),
    }

def update(accepted):
    return {
        "accepted": "1" if accepted else "0",
        "innovation_x": "0.0",
        "innovation_y": "0.0",
        "innovation_yaw": "0.0",
    }


class AnalyzeStackReplayTest(unittest.TestCase):
    def test_angle_wrap(self):
        self.assertAlmostEqual(MODULE.angle_diff(math.pi - 0.01, -math.pi + 0.01), 0.02)

    def test_gate_disagreement_and_peak(self):
        keys = [(i * 100_000_000, f"{i:08x}") for i in range(8)]
        left = {key: frame(i, 0.0, 0.0) for i, key in enumerate(keys)}
        right = {key: frame(i, 0.0, 0.0) for i, key in enumerate(keys)}
        right[keys[1]]["quality_pass"] = False
        left[keys[1]]["ekf_update"] = update(True)
        right[keys[1]]["ekf_update"] = update(False)
        right[keys[2]]["raw"] = pose(0.4)
        right[keys[2]]["ekf_output"] = pose(0.08)
        result = MODULE.compare(left, right, "test")
        self.assertEqual(result["quality_gate_disagreements"], 1)
        self.assertEqual(result["ekf_gate_disagreements"], 1)
        self.assertAlmostEqual(result["max_ndt_event"]["ndt_translation"], 0.4)
        self.assertAlmostEqual(result["max_ndt_event"]["attenuation_ratio"], 0.2)
        self.assertIsNotNone(result["recovery_primary"])

    def test_recovery_requires_five_consecutive_frames(self):
        rows = []
        values = [0.4, 0.09, 0.08, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05]
        for index, value in enumerate(values):
            rows.append(
                {
                    "index": index,
                    "original_ns": index * 100_000_000,
                    "ekf_translation": value,
                    "ekf_yaw": 0.0,
                }
            )
        recovered = MODULE.recovery(rows, 0, 0.10, 0.01)
        self.assertEqual(recovered["frames"], 5)
        self.assertAlmostEqual(recovered["seconds"], 0.8)


if __name__ == "__main__":
    unittest.main()
