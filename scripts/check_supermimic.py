"""GPU smoke test: environment stepping, state and actor history parity.

python scripts/check_supermimic.py --task Lite3-Chase-Flat --num-envs 8
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='Lite3-Chase-Flat')
    ap.add_argument('--num-envs', type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('This environment smoke test requires an NVIDIA CUDA GPU.')
    import lite3_supermimic.env
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from lite3_supermimic.supermimic import state as st
    from lite3_supermimic.robot.constants import JOINT_NAMES
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.commands['target'].debug_vis = False
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        obs, _ = env.reset()
        assert tuple(env.scene['robot'].joint_names) == JOINT_NAMES
        consts = st.robot_consts(env)
        assert torch.allclose(consts.action_scale, torch.full_like(consts.action_scale, 0.25))
        term = env.command_manager.get_term('target')
        terms = st.obs_terms(cfg)
        h = cfg.observations['actor'].history_length
        action = torch.zeros(args.num_envs, 12, device=env.device)
        for _ in range(h + 2):
            previous = obs['actor'].clone()
            obs, _, terminated, truncated, _ = env.step(action)
            s = st.robot_state(env, term.height_field)
            pose = st.planar_pose(env)
            lidar = term.lidar() if term.cfg.target.obstacles.enabled else None
            scan = term.scan() if term.cfg.terrain else None
            frame = st.obs_frame(s, action, term.command,
                                 env.scene['robot'].data.encoder_bias, noise=False,
                                 lidar=lidar, height_scan=scan)
            expected = st.push_history(previous, frame, h, terms)
            keep = ~(terminated | truncated)
            assert keep.any(), 'All environments terminated during smoke test'
            torch.testing.assert_close(obs['actor'][keep], expected[keep], atol=1e-4, rtol=1e-4)
            assert s.shape == (args.num_envs, st.STATE_DIM)
            assert torch.isfinite(s).all() and torch.isfinite(pose).all()
        print(f'{args.task}: state, actions, and imagined/real observation history agree.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
