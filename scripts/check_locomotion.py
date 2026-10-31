"""Smoke-check real environment history/rewards and a differentiable model rollout."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lite3_supermimic.locomotion.config import HISTORY, STAGES, WEIGHTS
from lite3_supermimic.locomotion.core import TERMS, observation
from lite3_supermimic.locomotion.imagine import Imagination
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.imagine import ImagineCfg
from lite3_supermimic.supermimic.train import Actor, Critic
from lite3_supermimic.supermimic.world_model import WorldModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=tuple(STAGES), default="flat")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compact-terrain", action="store_true", help="Small/coarse map for CPU diagnostics only")
    args = ap.parse_args()
    from lite3_supermimic.locomotion.env import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    cfg = make_env_cfg(args.stage, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.commands["velocity"].debug_vis = False
    if args.compact_terrain and STAGES[args.stage].terrain:
        gen = cfg.scene.terrain.terrain_generator
        gen.num_rows, gen.num_cols, gen.size = 2, 2, (6.0, 6.0)
        gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in ("stairs", "rough")}
        for tile in gen.sub_terrains.values():
            tile.proportion = 0.5
        cfg.commands["velocity"].field_res = 0.1
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        term = env.command_manager.get_term("velocity")
        a = torch.zeros(args.num_envs, 12, device=env.device)
        for _ in range(HISTORY + 3):
            previous = obs["actor"].clone()
            obs, rewards, terminated, truncated, _ = env.step(a)
            s, p = st.robot_state(env, term.height_field), st.planar_pose(env)
            frame = observation(s, a, term.command, env.scene["robot"].data.encoder_bias, term.scan())
            expected = st.push_history(previous, frame, HISTORY, TERMS)
            keep = ~(terminated | truncated)
            assert keep.any(), "All environments terminated"
            torch.testing.assert_close(obs["actor"][keep], expected[keep], atol=1e-4, rtol=1e-4)
            assert torch.isfinite(rewards).all() and torch.isfinite(s).all()
        n, obs_dim = obs["actor"].shape
        consts = st.robot_consts(env)
        imag = Imagination(ImagineCfg(horizon=3, dt=env.step_dt, weights=WEIGHTS),
                           consts, STAGES[args.stage], HISTORY, env.device, term.height_field)
        wm = WorldModel((32, 32)).to(env.device)
        actor = Actor(obs_dim, (32, 32), 12).to(env.device)
        critic = Critic(imag.critic_dim, (32, 32)).to(env.device)
        for net in (wm, critic):
            net.requires_grad_(False)
        for param in wm.net.parameters():
            param.data.mul_(0.01)
        seed = dict(env=torch.arange(n, device=env.device), s=s, pose=p, a_prev=a,
                    obs=obs["actor"], gen=term.packed(), xi=st.ContextReader(env).read(),
                    enc_bias=env.scene["robot"].data.encoder_bias)
        objective, _ = imag.rollout(wm, actor, critic, seed)
        (-objective.mean()).backward()
        grads = [param.grad for param in actor.parameters() if param.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
        print(f"{args.stage}: environment/history/finite rewards and imagined actor gradients passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
