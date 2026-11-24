# Copyright (c) 2026-2027 zh
"""
内容：
    用同一个 Lite3 检查点回放原生楼梯和石块，并在独立场景测试坡道及随机凹凸。
用法：
    python tools/play_lite3_terrains.py /path/to/model_1500.pt --vx 0.60
"""
import argparse
import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.terrains.config import hf_pyramid_slope, hf_pyramid_slope_inv, random_rough
from mjlab.viewer import NativeMujocoViewer
from tools.eval_lite3_stairs import reset_on, STEERING_MODES

from src.tasks.locomotion.go2_supermimic.mdp import terrains
from src.tasks.locomotion.go2_supermimic.rl.playback import SuperMimicPlayEnv, make_playback
from src.tasks.locomotion.go2_supermimic.rl.transition_adapter import Go2TransitionAdapter


NATIVE = (
    ("ascending stairs", "pyramid_stairs_inv_3", 5, 0.575),
    ("descending stairs", "pyramid_stairs_3", 3, 0.65),
    ("uneven blocks", "box_random_grid", 3, 0.65),
)
EXTRA = (
    ("uphill slope", "preview_slope_up", 3, 0.65),
    ("downhill slope", "preview_slope_down", 3, 0.65),
    ("gravel-like heightfield", "preview_gravel", 3, 0.0),
)
REFERENCE_GENERATOR = terrains.reference_generator


def preview_generator(cfg):
    generator = REFERENCE_GENERATOR(cfg)
    generator.sub_terrains = {
        "preview_slope_up": hf_pyramid_slope_inv(proportion=1.0, slope_range=(.15, .35)),
        "preview_slope_down": hf_pyramid_slope(proportion=1.0, slope_range=(.15, .35)),
        "preview_gravel": random_rough(proportion=1.0, noise_range=(.01, .045), noise_step=.005),
    }
    generator.num_cols = len(generator.sub_terrains)
    return generator


def show(env, policy, cases, args):
    for name, family, level, start in cases:
        print(f"\n[INFO] {name} | {family} L{level} | x={start:g} | vx={args.vx:g} m/s | {args.steering}", flush=True)
        reset_on(env, family, start, level, vx=args.vx, steering=args.steering)
        NativeMujocoViewer(env, policy).run(catch_sigint=False)


def main():
    parser = argparse.ArgumentParser(description="Preview one Lite3 checkpoint on stairs, slopes and rough ground.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vx", type=float, default=.60)
    parser.add_argument("--steering", choices=STEERING_MODES, default="heading")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not math.isfinite(args.vx) or args.vx <= 0.:
        parser.error("Forward command must be positive and finite.")

    env, policy = make_playback("Lite3-SuperMimic-Rough", checkpoint_file=args.checkpoint,
                                num_envs=1, device=args.device)
    try:
        show(env, policy, NATIVE, args)
        cfg = env.adapter.cfg
    finally:
        env.close()

    # 权重已按原训练地形验证。坡道用同一观测和动作协议装配独立测试场景。
    with patch.object(terrains, "reference_generator", preview_generator):
        test_env = SuperMimicPlayEnv(Go2TransitionAdapter(cfg, device=args.device, play=True))
    try:
        show(test_env, policy, EXTRA, args)
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
