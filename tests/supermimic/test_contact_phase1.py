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
    def test_geometry_grid_reconstruction(self):
        from lite3_supermimic.locomotion.geometry_compare import ray_grid_interpolate, error_stats
        points = np.array([[.013, .017], [-.029, .035], [.105, -.04]])
        plane = lambda xy: xy[:, 0] * .3 + xy[:, 1] * .2 + .7
        for resolution in (.1, .05, .02):
            values, _, _ = ray_grid_interpolate(points, (-.5, -.5), resolution, plane)
            np.testing.assert_allclose(values, plane(points), atol=1e-12)
        # A vertical step inside a coarse cell is smeared; finer cells can recover it.
        step = lambda xy: np.where(xy[:, 0] >= .05, .18, 0.)
        point = np.array([[.075, .013]])
        coarse, _, _ = ray_grid_interpolate(point, (0., 0.), .1, step)
        fine, _, _ = ray_grid_interpolate(point, (0., 0.), .02, step)
        self.assertGreater(abs(coarse[0] - .18), .04)
        self.assertAlmostEqual(fine[0], .18)
        stats = error_stats(np.array([0., .2, np.nan]), np.array([.1, .1, .1]))
        self.assertEqual(stats['invalid'], 1)
        self.assertEqual(stats['underestimate_over_5cm_fraction'], .5)
        self.assertEqual(stats['overestimate_over_5cm_fraction'], .5)

    def test_geometry_common_support_and_original_nodes(self):
        from lite3_supermimic.locomotion.geometry_compare import common_support_comparison, cached_node_audit
        from lite3_supermimic.target.terrain import HeightField
        result = common_support_comparison({'a': np.array([0., 1., np.nan]),
                                            'b': np.array([0., np.nan, 3.])},
                                           np.array([0., 1., 3.]), np.array([True, True, False]))
        self.assertEqual(result['common_valid_points'], 1)
        self.assertEqual(result['methods']['a']['all']['samples'], result['methods']['b']['all']['samples'])
        field = HeightField(torch.ones(5, 600), (-54.5, -6.5), .1)
        corners = np.array([[[-54.5, -6.5], [-4.5, -6.4]]])
        oracle = lambda xy: np.ones(len(xy))
        report, rows = cached_node_audit(field, corners, oracle)
        self.assertEqual(report['stable_nodes'], 2)
        self.assertEqual(report['cached_vs_original_arange']['mae_m'], 0.)
        self.assertEqual(rows[1, 2], np.arange(-54.5, 5.5, .1)[500])
        step = lambda xy: np.where(xy[:, 0] < -4.5, 0., .18)
        report, _ = cached_node_audit(field, corners, step)
        self.assertEqual(report['edge_sensitive_or_missing_nodes'], 1)
        missing = lambda xy: np.full(len(xy), np.nan)
        report, _ = cached_node_audit(field, corners, missing)
        self.assertEqual(report['original_arange_no_hits'], 2)
        self.assertEqual(report['cached_vs_original_arange']['samples'], 0)
        self.assertEqual(report['cached_vs_original_builder_zero_fallback']['samples'], 2)

    def test_terrain_reencoding_and_ray_retry(self):
        from lite3_supermimic.locomotion.terrain_reencode import reencode_chunk, build_diagnostic_grid
        from lite3_supermimic.target.terrain import HeightField
        old = HeightField(torch.zeros(40,40), (-1.,-1.), .1)
        new = HeightField(torch.full((40,40), .18), (-1.,-1.), .1)
        state = torch.randn(2,3,70)
        chunk = dict(s0=state, s1=state.clone(), pose0=torch.zeros(2,3,3), pose1=torch.zeros(2,3,3),
                     action=torch.zeros(2,3,12), contact=torch.zeros(2,3,8,dtype=torch.bool), relief=torch.zeros(2,3,77))
        converted = reencode_chunk(chunk,old,new)
        torch.testing.assert_close(converted['s0'][...,0]+.18, state[...,0])
        torch.testing.assert_close(converted['s0'][...,1:],state[...,1:])
        self.assertIs(converted['action'],chunk['action'])
        self.assertIs(converted['contact'],chunk['contact'])
        torch.testing.assert_close(converted['relief'],torch.zeros_like(converted['relief']))
        calls = [0]
        def ray(xy):
            calls[0] += 1
            return np.full(len(xy), np.nan if calls[0]==1 else .18)
        field, report = build_diagnostic_grid((0.,0.,.04,.04),.02,ray)
        self.assertGreater(report['no_hit_nodes_retried'],0)
        torch.testing.assert_close(field.h,torch.full_like(field.h,.18))
        with self.assertRaisesRegex(ValueError,'Unresolved'):
            build_diagnostic_grid((0.,0.,.04,.04),.02,lambda xy:np.full(len(xy),np.nan))
        with self.assertRaisesRegex(ValueError,'oversized'):
            build_diagnostic_grid((0.,0.,100.,100.),.02,ray,max_nodes=100)

    def test_same_prefix_counterfactual_common_population(self):
        from scripts.audit_action_response import summarize_counterfactual
        active=torch.tensor([True,True,False])
        base=dict(active=active,done=torch.tensor([False,False,False]),
                  prob=torch.tensor([.8,.6,.1]),actual=torch.tensor([True,False,False]),
                  foot=torch.tensor([False,True,True]),metadata={'step':20})
        changed=dict(active=active,done=torch.tensor([False,True,False]),
                     prob=torch.tensor([.2,.9,.1]),actual=torch.tensor([False,True,False]),
                     foot=torch.tensor([True,False,True]),metadata={'step':20})
        report=summarize_counterfactual({'baseline':base,'hipy_minus':changed})
        self.assertEqual(report['baseline']['common_valid'],1)
        self.assertEqual(report['hipy_minus']['common_valid'],1)
        self.assertAlmostEqual(report['hipy_minus']['actual_risk_delta_vs_baseline'],-1.)
        self.assertAlmostEqual(report['hipy_minus']['predicted_risk_delta_vs_baseline'],-.6,places=6)
        with self.assertRaisesRegex(ValueError,'Active masks'):
            summarize_counterfactual({'baseline':base,'hipy_minus':dict(changed,active=~active)})
        with self.assertRaisesRegex(ValueError,'No common valid'):
            summarize_counterfactual({'baseline':dict(base,done=active),'hipy_minus':changed})

    def test_action_response_prefix_guard_reports_field(self):
        from scripts.audit_action_response import verify_same_prefix
        pre={'state':torch.tensor([1.,2.]),'foot_contact':torch.tensor([True,False])}
        base={'step':7,'active':torch.tensor([True,False]),'pre':pre}
        close={'step':7,'active':base['active'].clone(),
               'pre':{'state':torch.tensor([1.000001,2.]),'foot_contact':pre['foot_contact'].clone()}}
        differences=verify_same_prefix({'baseline':base,'hipy_minus':close})
        self.assertLess(differences['hipy_minus']['state'],1e-5)
        changed={'step':7,'active':base['active'].clone(),
                 'pre':{'state':torch.tensor([1.1,2.]),'foot_contact':pre['foot_contact'].clone()}}
        with self.assertRaisesRegex(RuntimeError,'hipy_minus.state: max abs difference'):
            verify_same_prefix({'baseline':base,'hipy_minus':changed})
        with self.assertRaisesRegex(RuntimeError,'trigger step'):
            verify_same_prefix({'baseline':base,'hipy_minus':dict(close,step=8)})

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
            for seed in (0, 1, 2, 3):
                r = ContactRecorder(root / str(seed), dict(seed=seed, checkpoint_sha256=digest,
                    stage='terrain', terrain='stairs_inv', terrain_level=8, command=[1.,0.,0.], dt=.02,
                    num_envs=3, seconds=.16))
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
            refined_path = root / "head/contact_world_model.pt"
            refined_hash = file_hash(refined_path)
            # Run both ablation arms end to end on a physically identical synthetic dataset.
            from lite3_supermimic.locomotion.terrain_reencode import reencode_chunk
            import shutil
            for split, seed in zip(('train','validation','test'), (0,1,2)):
                old_dir = root/'coarse'/split/'up8_v10'
                old_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(root/str(seed),old_dir)
                fine_dir = root/'fine'/split/'up8_v10'
                fine_dir.mkdir(parents=True)
                old_field = HeightField.load(str(old_dir/'field.pt'),'cpu')
                fine_field = HeightField(old_field.h + .04, old_field.origin, old_field.res)
                fine_field.save(str(fine_dir/'field.pt'))
                manifest = json.loads((old_dir/'manifest.json').read_text())
                hashes = {}
                for filename in manifest['files']:
                    chunk = torch.load(old_dir/filename,weights_only=True)
                    torch.save(reencode_chunk(chunk,old_field,fine_field),fine_dir/filename)
                    hashes[filename] = file_hash(old_dir/filename)
                manifest['representation_change'] = dict(source_directory=str(old_dir),
                    source_manifest_sha256=file_hash(old_dir/'manifest.json'),source_chunks_sha256=hashes)
                (fine_dir/'manifest.json').write_text(json.dumps(manifest))
            ablation = script.parent/'fit_terrain_ablation.py'
            argv = [str(ablation),'--source',str(refined_path),'--old-root',str(root/'coarse'),
                    '--fine-root',str(root/'fine'),'--output-dir',str(root/'ablation'),
                    '--updates','2','--batch','8','--eval-windows','12','--eval-interval','1','--device','cpu',
                    '--arms','coarse','fine','hybrid']
            with patch.object(sys,'argv',argv): runpy.run_path(str(ablation),run_name='__main__')
            summary = json.loads((root/'ablation/summary.json').read_text())
            self.assertFalse(summary['actor_updated'])
            self.assertIn('base_world_z_rmse',summary['comparison']['step_5']['fine'])
            self.assertIn('base_world_z_rmse',summary['comparison']['step_5']['hybrid'])
            from scripts.fit_terrain_ablation import CheckedDataset, HybridDataset
            old_ds=CheckedDataset([root/'coarse/train/up8_v10'],5)
            fine_ds=CheckedDataset([root/'fine/train/up8_v10'],5)
            hybrid=HybridDataset(old_ds,fine_ds)
            indices=torch.arange(3)
            old_batch=old_ds.batch(indices,'cpu')
            hybrid_batch=hybrid.batch(indices,'cpu')
            for key in old_batch:
                if key!='relief': self.assertTrue(torch.equal(old_batch[key],hybrid_batch[key]))
            torch.testing.assert_close(hybrid_batch['relief'],old_batch['relief']-.04)
            pose=old_batch['pose0'][:,0]
            torch.testing.assert_close(hybrid.ground_at(indices,pose,'cpu'),old_ds.ground_at(indices,pose,'cpu'))
            # Same physical input height, fine surface, old height reference.
            expected=old_batch['s0'][:,0,0,None]-.04
            torch.testing.assert_close(hybrid_batch['s0'][:,0,0,None]+hybrid.relief_at(indices,pose,'cpu'),
                                       expected.expand(-1,77))
            fine_ds.fields[0].h=old_ds.fields[0].h.clone()
            torch.testing.assert_close(hybrid.relief_at(indices,pose,'cpu'),old_ds.relief_at(indices,pose,'cpu'))
            with self.assertRaisesRegex(ValueError,'left stored field'):
                hybrid.relief_at(indices,pose+100,'cpu')

            self.assertEqual(file_hash(refined_path),refined_hash)

            argv = [str(audit), "--model", str(refined_path), "--fresh-data", str(root / "3"),
                    "--output", str(root / "fresh.json"), "--device", "cpu"]
            with patch.object(sys, "argv", argv):
                runpy.run_path(str(audit), run_name="__main__")
            fresh = json.loads((root / "fresh.json").read_text())
            self.assertTrue(fresh["fresh_seed_audit"])
            self.assertEqual(fresh["provenance"]["fresh_0_3"][0]["metadata"]["seed"], 3)
            self.assertEqual(file_hash(refined_path), refined_hash)
            from scripts.audit_contact_probabilities import fresh_datasets
            with self.assertRaisesRegex(ValueError, "unused seed"):
                fresh_datasets(refined["args"], [root / "1"], digest)
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                fresh_datasets(refined["args"], [root / "3"], "wrong")


if __name__ == "__main__":
    unittest.main()
