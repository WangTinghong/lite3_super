"""Locate terrain representation errors on saved trajectories; no training or policy edits."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True, help='One completed contact collection directory')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--samples', type=int, default=128, help='Recorded robot states, each with 16 path points')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    dest = Path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):
        p.error('output-dir must be new or empty')
    if not args.worker:
        from scripts.contact_phase1 import child_environment
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                        *sys.argv[1:], '--worker'], env=child_environment(), check=True)
        return

    import numpy as np
    import torch
    import mujoco
    from importlib.metadata import version
    from lite3_supermimic.locomotion.contact_data import ContactDataset, file_hash
    from lite3_supermimic.locomotion.core import feet_world, validate_checkpoint
    from lite3_supermimic.locomotion.geometry_compare import ray_grid_interpolate, error_stats, common_support_comparison, cached_node_audit
    from lite3_supermimic.locomotion.env import make_env_cfg
    from lite3_supermimic.target.terrain import HeightField
    from mjlab.envs import ManagerBasedRlEnv

    directory = Path(args.data)
    manifest = json.loads((directory / 'manifest.json').read_text())
    meta = manifest['metadata']
    if file_hash(args.checkpoint) != meta['checkpoint_sha256']:
        p.error('checkpoint differs from collection source')
    for package in ('mjlab', 'mujoco', 'mujoco-warp'):
        if version(package) != meta['versions'][package]:
            p.error(f'{package} version differs from collection; reconstruct with matching versions')
    field = HeightField.load(str(directory / 'field.pt'), 'cpu')
    ds = ContactDataset([directory], horizon=1)
    ids = torch.randperm(len(ds.index), generator=torch.Generator().manual_seed(42))[:args.samples]
    b = ds.batch(ids, 'cpu')
    state, pose = b['s0'][:, 0], b['pose0'][:, 0]
    feet = feet_world(state, pose, field)[..., :2]
    offsets = torch.tensor([0., .05, .12, .25])
    heading = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
    xy = (feet[:, :, None] + heading[:, None, None] * offsets[None, None, :, None]).reshape(-1, 2)
    cached = field.sample(xy).numpy()
    points = xy.numpy().astype(np.float64)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt)
    cfg = make_env_cfg(meta['stage'], play=True, harder_stairs=bool(ckpt['cfg'].get('harder_stairs', False)))
    cfg.seed = meta['seed']
    cfg.scene.num_envs = meta['num_envs']
    cfg.auto_reset = False
    cfg.commands['velocity'].fixed_command = tuple(meta['command'])
    cfg.commands['velocity'].fixed_level = meta['terrain_level']
    generator = cfg.scene.terrain.terrain_generator
    if generator is None or meta['terrain'] not in generator.sub_terrains:
        p.error('requires a terrain collection with one specified terrain type')
    tile = generator.sub_terrains[meta['terrain']]
    tile.proportion = 1.0
    generator.sub_terrains = {meta['terrain']: tile}
    print('Reconstructing source terrain; no policy rollout or optimizer updates.', flush=True)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        exclude = {int(g) for g in env.scene['robot'].indexing.geom_ids}
        geoms = [g for g in range(model.ngeom) if g not in exclude]
        saved = model.geom_group.copy()
        model.geom_group[:] = 5
        model.geom_group[geoms] = 0
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        gid = np.zeros(1, dtype=np.int32)
        def ray_heights(coords):
            values = np.full(len(coords), np.nan)
            for i, (x, y) in enumerate(coords):
                d = mujoco.mj_ray(model, data, np.array([x, y, 20.]),
                                  np.array([0., 0., -1.]), group, 1, -1, gid)
                if d >= 0:
                    values[i] = 20. - d
            return values
        try:
            exact = ray_heights(points)
            predictions = {'cached_grid': cached}
            for resolution in (field.res, .05, .02):
                print(f'Auditing grid {resolution:g} m at {len(points)} path points', flush=True)
                values, corners, heights = ray_grid_interpolate(points, field.origin, resolution, ray_heights)
                predictions[f'raycast_grid_{resolution:g}m'] = values
                if resolution == field.res:
                    original_corners = corners
            print('Checking unique cached nodes and +/-1 micrometre edge sensitivity', flush=True)
            node_report, node_rows = cached_node_audit(field, original_corners, ray_heights)
        finally:
            model.geom_group[:] = saved
        paths = exact.reshape(-1, 4)
        edge = np.repeat(np.isfinite(paths).all(-1) & (np.ptp(paths, axis=-1) > .04), 4)
        grid_xy = (points - np.asarray(field.origin)) / field.res
        outside = ((grid_xy[:, 0] < 0) | (grid_xy[:, 1] < 0)
                   | (grid_xy[:, 0] > field.h.shape[1] - 1.001)
                   | (grid_xy[:, 1] > field.h.shape[0] - 1.001))
        result = dict(points_outside_cached_grid=int(outside.sum()), actor_updated=False, model_updated=False, policy_inputs_changed=False,
                      checkpoint_sha256=meta['checkpoint_sha256'], collection=meta,
                      sampled_states=len(ids), grid_origin=field.origin, grid_resolution_m=field.res,
                      audit_version=2, cached_node_checks=node_report,
                      common_support=common_support_comparison(predictions, exact, edge),
                      comparison={name: {'all': error_stats(v, exact),
                                         'edge_paths': error_stats(v[edge], exact[edge])}
                                  for name, v in predictions.items()},
                      note='Use common_support for grid comparisons. Node checks reproduce original arange coordinates and audit edge sensitivity. No-hit zero fill is only an explicitly labelled builder-compatibility check, not ground truth. No grid is installed.')
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / 'samples.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state_sample', 'chunk', 'time_index', 'env_index', 'leg', 'ahead_m',
                             'x', 'y', 'ray_height_m', 'edge_path', *predictions])
            for i, (x, y) in enumerate(points):
                sample, leg, offset = i // 16, (i // 4) % 4, i % 4
                writer.writerow([sample, *ds.index[ids[sample]].tolist(), ('FR','FL','HR','HL')[leg],
                                 float(offsets[offset]), x, y, exact[i], bool(edge[i]),
                                 *[v[i] for v in predictions.values()]])
        np.savetxt(dest / 'nodes.csv', node_rows, delimiter=',', comments='',
                   header='grid_x,grid_y,original_x,original_y,cached_z,nominal_ray_z,original_ray_z,neighbor_height_span,edge_sensitive_or_missing')
        (dest / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        print(f'Saved {dest / "summary.json"} and samples.csv', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
