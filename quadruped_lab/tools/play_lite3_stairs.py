# Copyright (c) 2026-2027 zh
"""
内容：
    在 MJLab 原生窗口回放 Lite3 检查点，固定楼梯等级、起点和前向速度。

用法：
    python tools/play_lite3_stairs.py /path/to/model_2500.pt --stair-level 5 --start-x 0.575 --vx 1.00
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.viewer import NativeMujocoViewer

from src.tasks.locomotion.go2_supermimic.rl.playback import make_playback
from tools.eval_lite3_stairs import STAIRS, reset_on


def main():
    parser = argparse.ArgumentParser(description="Play a Lite3 checkpoint on a fixed stair tile in MJLab.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint with task_config.json beside it.")
    parser.add_argument("--stair-level", type=int, default=5)
    parser.add_argument("--start-x", type=float, default=0.575, help="Stair tile X offset in meters.")
    parser.add_argument("--vx", type=float, default=1.00, help="Forward command in m/s.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stair_level < 0 or not math.isfinite(args.start_x) or not math.isfinite(args.vx) or args.vx <= 0.0:
        parser.error("Expected a nonnegative stair level, finite start offset and positive forward command.")

    env, policy = make_playback("Lite3-SuperMimic-Rough", checkpoint_file=args.checkpoint, num_envs=1, device=args.device)
    try:
        if args.stair_level >= env.adapter.map.rows:
            parser.error(f"Stair level must be between 0 and {env.adapter.map.rows - 1}.")
        reset_on(env, STAIRS, args.start_x, args.stair_level, vx=args.vx)
        NativeMujocoViewer(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
