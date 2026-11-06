"""Create separate 2cm offline datasets. Never alter a policy or its cached height map."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='audit/contact_phase1/data')
    p.add_argument('--output-dir', default='audit/terrain_2cm_pilot/data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.worker:
        from scripts.contact_phase1 import child_environment
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=child_environment(), check=True)
        return
    import numpy as np
    import torch
    import mujoco
    from importlib.metadata import version
    from mjlab.envs import ManagerBasedRlEnv
    from lite3_supermimic.locomotion.env import make_env_cfg
    from lite3_supermimic.locomotion.contact_data import file_hash
    from lite3_supermimic.locomotion.core import validate_checkpoint
    from lite3_supermimic.locomotion.terrain_reencode import reencode_chunk, build_diagnostic_grid
    from lite3_supermimic.target.terrain import HeightField
    source_hash = file_hash(a.checkpoint)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt)
    for split in ('train', 'validation', 'test'):
        src = Path(a.data_root) / split / 'up8_v10'
        out = Path(a.output_dir) / split / 'up8_v10'
        manifest = json.loads((src / 'manifest.json').read_text())
        meta = manifest['metadata']
        if meta['checkpoint_sha256'] != source_hash or meta['terrain'] != 'stairs_inv' or meta['terrain_level'] != 8:
            raise ValueError(f'Unexpected source collection: {src}')
        for package in ('mjlab', 'mujoco', 'mujoco-warp'):
            if meta['versions'][package] != version(package):
                raise ValueError(f'{package} version differs from collection')
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output must be empty: {out}; preserve partial work and use a new --output-dir')
        old = HeightField.load(str(src / 'field.pt'), 'cpu')
        lows, highs = [], []
        for name in manifest['files']:
            chunk = torch.load(src / name, map_location='cpu', weights_only=True)
            xy = torch.cat([chunk[k][..., :2].reshape(-1, 2) for k in ('pose0', 'pose1')])
            lows.append(xy.amin(0)); highs.append(xy.amax(0))
        # Covers all recorded poses plus scan extent and a generous rollout margin.
        lo = torch.stack(lows).amin(0).numpy() - 2.
        hi = torch.stack(highs).amax(0).numpy() + 2.
        bounds = (*lo.tolist(), *hi.tolist())
        cfg = make_env_cfg(meta['stage'], play=True, harder_stairs=bool(ckpt['cfg'].get('harder_stairs', False)))
        cfg.seed = meta['seed']; cfg.scene.num_envs = meta['num_envs']; cfg.auto_reset = False
        cfg.commands['velocity'].fixed_level = meta['terrain_level']
        cfg.commands['velocity'].fixed_command = tuple(meta['command'])
        generator = cfg.scene.terrain.terrain_generator
        tile = generator.sub_terrains[meta['terrain']]; tile.proportion = 1.
        generator.sub_terrains = {meta['terrain']: tile}
        print(f'{split}: reconstructing physical terrain, actor not loaded', flush=True)
        env = ManagerBasedRlEnv(cfg, device=a.device)
        try:
            model = env.sim.mj_model; data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            excluded = {int(g) for g in env.scene['robot'].indexing.geom_ids}
            geoms = [g for g in range(model.ngeom) if g not in excluded]
            groups_before = model.geom_group.copy()
            model.geom_group[:] = 5; model.geom_group[geoms] = 0
            group = np.array([1,0,0,0,0,0], dtype=np.uint8); gid = np.zeros(1,dtype=np.int32)
            def ray(coords):
                values = np.full(len(coords), np.nan)
                for i, (x,y) in enumerate(coords):
                    d = mujoco.mj_ray(model,data,np.array([x,y,20.]),np.array([0.,0.,-1.]),group,1,-1,gid)
                    if d >= 0: values[i] = 20. - d
                return values
            try:
                fine, grid_report = build_diagnostic_grid(bounds,.02,ray)
            finally:
                model.geom_group[:] = groups_before
        finally:
            env.close()
        out.mkdir(parents=True, exist_ok=True)
        fine.save(str(out / 'field.pt'))
        hashes = {}
        for name in manifest['files']:
            chunk = torch.load(src / name, map_location='cpu', weights_only=True)
            converted = reencode_chunk(chunk,old,fine)
            torch.save(converted,out / name)
            hashes[name] = file_hash(src / name)
        updated = dict(manifest)
        updated['metadata'] = dict(meta, field_res_m=.02)
        updated['representation_change'] = dict(source_directory=str(src.resolve()),
            source_manifest_sha256=file_hash(src / 'manifest.json'), source_chunks_sha256=hashes,
            actor_updated=False, old_policy_observations_used_for_collection=True,
            transformed_fields=['s0.base_height','s1.base_height','relief'],
            near_mask='preserved from original for paired evaluation',grid=grid_report)
        # Original ray audit describes the original field, not the new grid.
        updated['original_terrain_audit'] = updated.pop('terrain_audit', {})
        (out / 'manifest.json').write_text(json.dumps(updated,indent=2)+'\n')
        print(f'Saved {out}; original data unchanged',flush=True)


if __name__ == '__main__': main()
