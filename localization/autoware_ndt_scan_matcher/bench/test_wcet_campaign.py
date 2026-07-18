#!/usr/bin/env python3

import argparse
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

SCRIPT = pathlib.Path(__file__).with_name("wcet_campaign.py")
SPEC = importlib.util.spec_from_file_location("wcet_campaign", SCRIPT)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)
CONFIG_PATH = pathlib.Path(__file__).with_name("campaign_config_unified.json")


class UnifiedCampaignTest(unittest.TestCase):
    def setUp(self):
        self.cfg = CAMPAIGN.load_config(CONFIG_PATH)

    def test_plan_has_ten_fixtures_and_five_equal_series(self):
        CAMPAIGN.validate_config(self.cfg)
        cells = CAMPAIGN.build_plan(self.cfg, 1)
        self.assertEqual(len(cells), 100)
        counts = {}
        for cell in cells:
            counts[cell["series"]] = counts.get(cell["series"], 0) + 1
            self.assertEqual(cell["samples"], 1000)
        self.assertEqual(
            counts,
            {
                "warm": 20,
                "cold": 20,
                "corunner:membw": 20,
                "corunner:llc": 20,
                "corunner:fp": 20,
            },
        )

    def test_pareto_paths_are_explicit_and_present(self):
        for name in ("pareto_01", "pareto_02"):
            path = CAMPAIGN.fixture_path(self.cfg, name)
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.name, "pareto")

    def test_completed_cell_requires_exact_sample_count_and_manifest(self):
        cell = next(item for item in CAMPAIGN.build_plan(self.cfg, 1)
                    if item["series"] == "warm")
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            output = root / "cell.json"
            sidecar = root / "cell.cell.json"
            fixture = {
                cell["engine"]: {
                    "iteration_num": 1,
                    "samples_ms": [1.0] * cell["samples"],
                }
            }
            output.write_text(json.dumps({
                "meta": {"iters": cell["samples"], "engine": cell["engine"]},
                "fixtures": {cell["fixture"]: fixture},
            }))
            sidecar.write_text(json.dumps({
                "cell": cell,
                "problems": [],
                "manifest": {
                    "campaign_config_hash": CAMPAIGN.config_hash(self.cfg),
                    "fixture_hashes": {
                        cell["fixture"]: CAMPAIGN.sha256_of(
                            CAMPAIGN.fixture_path(self.cfg, cell["fixture"]))
                    },
                },
            }))
            self.assertIsNone(
                CAMPAIGN.completed_cell_problem(self.cfg, output, sidecar, cell))
            document = json.loads(output.read_text())
            document["meta"]["iters"] -= 1
            output.write_text(json.dumps(document))
            self.assertIn(
                "sample metadata",
                CAMPAIGN.completed_cell_problem(self.cfg, output, sidecar, cell),
            )

    def test_session_lock_rejects_a_different_boot(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = json.loads(json.dumps(self.cfg))
            cfg["output_dir"] = temp
            with mock.patch.object(CAMPAIGN, "boot_id", return_value="boot-a"):
                CAMPAIGN.ensure_session_lock(cfg, 1)
            with mock.patch.object(CAMPAIGN, "boot_id", return_value="boot-b"):
                with self.assertRaisesRegex(ValueError, "may not span boots"):
                    CAMPAIGN.ensure_session_lock(cfg, 1)

    def test_timing_campaign_rejects_traced_or_unknown_builds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            cache = root / "build" / CAMPAIGN.PKG / "CMakeCache.txt"
            cache.parent.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "NDT_BUILD_TRACED=OFF"):
                CAMPAIGN.require_untraced_timing_build(root)
            cache.write_text("NDT_BUILD_TRACED:BOOL=ON\n")
            with self.assertRaisesRegex(ValueError, "CMake cache: ON"):
                CAMPAIGN.require_untraced_timing_build(root)
            cache.write_text("NDT_BUILD_TRACED:BOOL=OFF\n")
            CAMPAIGN.require_untraced_timing_build(root)

    def test_status_reports_an_empty_session(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = json.loads(json.dumps(self.cfg))
            cfg["output_dir"] = temp
            args = argparse.Namespace(session=1)
            self.assertEqual(CAMPAIGN.cmd_status(cfg, args), 1)


if __name__ == "__main__":
    unittest.main()
