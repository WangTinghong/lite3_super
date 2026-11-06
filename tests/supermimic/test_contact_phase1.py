"""CPU tests: reset boundaries, held-out splits, shared gradients and labels."""
import json
import tempfile
import unittest
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pathlib import Path

import torch

from lite3_supermimic.locomotion.contact_data import ContactRecorder, ContactDataset, check_splits, window_starts, file_hash
from lite3_supermimic.locomotion.contact_model import ContactWorldModel, labels, classification_metrics, probability_diagnostics
from lite3_supermimic.supermimic.world_model import WorldModel


class ContactPhaseTests(unittest.TestCase):
    def test_probability_ranking_and_ties(self):
        truth = torch.tensor([True, False, True, False])
        tied = probability_diagnostics(torch.ones(4) * .1, truth)
        self.assertEqual(tied["average_precision"], .5)
        perfect = probability_diagnostics(torch.tensor([.3, .1, .2, .05]), truth)
        self.assertEqual(perfect["average_precision"], 1.0)
        self.assertEqual(classification_metrics(torch.zeros(4, 12), torch.zeros(4, 12))["foot_FR"]["average_precision"], None)

    def test_geometry_ray_misses_and_group_restoration(self):
        from lite3_supermimic.locomotion.terrain_audit import TerrainAudit
        from lite3_supermimic.target.terrain import HeightField
        audit = TerrainAudit.__new__(TerrainAudit)
        audit.model = SimpleNamespace(geom_group=np.array([2, 3]))
        audit.data, audit.geoms = None, [0]
        audit.errors, audit.near_errors = [], []
        audit.edge_count = audit.missed_edges = audit.no_hits = 0
        field = HeightField(torch.zeros(20, 20), (-1., -1.), .1)
        ray = SimpleNamespace(mj_ray=lambda *args: -1.0)
        with patch.dict(sys.modules, {"mujoco": ray}):
            audit.update(torch.zeros(1, 70), torch.zeros(1, 3), field)
            self.assertEqual(audit.result()["no_ray_hits"], 16)
            self.assertEqual(audit.result()["all"]["samples"], 0)
            np.testing.assert_array_equal(audit.model.geom_group, [2, 3])
            ray.mj_ray = lambda *args: 20.0
            audit.update(torch.zeros(1, 70), torch.zeros(1, 3), field)
            self.assertEqual(audit.result()["all"]["mae_m"], 0.0)
            with patch.object(ray, "mj_ray", side_effect=RuntimeError("ray failure")):
                with self.assertRaises(RuntimeError):
                    audit.update(torch.zeros(1, 70), torch.zeros(1, 3), field)
            np.testing.assert_array_equal(audit.model.geom_group, [2, 3])

    def test_reset_and_chunk_boundaries(self):
        done = torch.zeros(8, 2, dtype=torch.bool)
        done[3, 0] = True
        starts = window_starts(done, 3)
        self.assertEqual(starts[starts[:, 1] == 0, 0].tolist(), [0, 4, 5])
        self.assertEqual(starts[starts[:, 1] == 1, 0].tolist(), list(range(6)))

    def test_baseline_outputs_and_contact_gradient(self):
        torch.manual_seed(0)
        base = WorldModel((16, 16))
        model = ContactWorldModel((16, 16))
        model.load_baseline(base.state_dict())
        args = [torch.randn(7, n) for n in (70, 12, 12, 4, 77)]
        a, b = base(*args), model(*args)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
        model.predict(*args)[2].square().mean().backward()
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.net[-1].weight.grad)

    def test_joint_contact_labels_and_undefined_metrics(self):
        c = torch.tensor([[0, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        y = labels(c)
        self.assertEqual(y[0, 8:].tolist(), [1, 0, 0, 0])
        report = classification_metrics(torch.zeros_like(y), y)
        self.assertIsNone(report["foot_FR"]["recall"])
        self.assertEqual(report["shank_only_FR"]["recall"], 0)

    def test_recorder_split_and_event_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = []
            for seed in (0, 1):
                path = Path(tmp) / str(seed)
                r = ContactRecorder(path, dict(seed=seed, checkpoint_sha256="same"), chunk_steps=4)
                for t in range(8):
                    contact = torch.zeros(2, 8, dtype=torch.bool)
                    contact[0, 4] = t == 2
                    r.add(s0=torch.full((2, 70), float(t)), done=torch.zeros(2, dtype=torch.bool),
                          contact=contact, contact0=torch.zeros_like(contact))
                r.finish({})
                ds.append(ContactDataset([path], horizon=3))
            check_splits(*ds)
            with self.assertRaises(ValueError):
                check_splits(ds[0], ds[0])
            batch = ds[0].batch(torch.arange(len(ds[0].index)), "cpu")
            torch.testing.assert_close(batch["s0"][:, 1, 0] - batch["s0"][:, 0, 0], torch.ones(8))
            self.assertEqual(int(ds[0].event.sum()), 2)
            with self.assertRaises(ValueError):
                ContactRecorder(Path(tmp) / "0", {})

    def test_offline_fit_preserves_source_and_writes_holdout_report(self):
        from lite3_supermimic.target.terrain import HeightField
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            torch.save(dict(cfg={"wm_hidden": [16, 16]},
                            world_model=WorldModel((16, 16)).state_dict()), source)
            digest = file_hash(source)
            for seed in (0, 1, 2):
                r = ContactRecorder(root / str(seed), dict(seed=seed, checkpoint_sha256=digest))
                HeightField(torch.zeros(80, 80), (-3., -3.), .1).save(str(root / str(seed) / "field.pt"))
                for _ in range(8):
                    state = torch.randn(3, 70) * .01
                    contact = torch.rand(3, 8) > .7
                    r.add(s0=state, s1=state + .001, pose0=torch.zeros(3, 3),
                          pose1=torch.zeros(3, 3), action=torch.zeros(3, 12), prev=torch.zeros(3, 12),
                          context=torch.zeros(3, 4), relief=torch.zeros(3, 77), delta=torch.zeros(3, 3),
                          contact0=~contact, contact=contact, done=torch.zeros(3, dtype=torch.bool),
                          near=torch.ones(3, dtype=torch.bool))
                r.finish({})
            script = Path(__file__).resolve().parents[2] / "scripts/fit_contact_world_model.py"
            argv = [str(script), "--checkpoint", str(source), "--train", str(root / "0"),
                    "--validation", str(root / "1"), "--test", str(root / "2"),
                    "--output-dir", str(root / "fit"), "--device", "cpu", "--updates", "2",
                    "--eval-interval", "1", "--batch", "8", "--eval-windows", "12"]
            with patch.object(sys, "argv", argv):
                runpy.run_path(str(script), run_name="__main__")
            report = json.loads((root / "fit/report.json").read_text())
            self.assertFalse(report["actor_updated"])
            self.assertEqual(file_hash(source), digest)
            self.assertEqual(report["contact_model_test"]["step_5"]["all"]["contacts"]["shank_only_FR"]["samples"], 12)
            artifact = root / "fit/contact_world_model.pt"
            artifact_hash = file_hash(artifact)
            audit = script.parent / "audit_contact_probabilities.py"
            argv = [str(audit), "--model", str(artifact), "--output", str(root / "audit.json"), "--device", "cpu"]
            with patch.object(sys, "argv", argv):
                runpy.run_path(str(audit), run_name="__main__")
            self.assertEqual(file_hash(artifact), artifact_hash)
            self.assertFalse(json.loads((root / "audit.json").read_text())["model_updated"])
            refine = script.parent / "refine_contact_head.py"
            argv = [str(refine), "--source", str(artifact), "--output-dir", str(root / "head"),
                    "--device", "cpu", "--updates", "2", "--eval-interval", "1", "--batch", "8"]
            with patch.object(sys, "argv", argv):
                runpy.run_path(str(refine), run_name="__main__")
            refined = torch.load(root / "head/contact_world_model.pt", weights_only=False)
            previous = torch.load(artifact, weights_only=False)
            for key, value in previous["world_model"].items():
                if not key.startswith("contact_head."):
                    self.assertTrue(torch.equal(value, refined["world_model"][key]))
            self.assertEqual(file_hash(artifact), artifact_hash)


if __name__ == "__main__":
    unittest.main()
