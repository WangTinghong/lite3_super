"""Compare a saved actor with small FR HipY/Knee interventions in MuJoCo."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audit/joint_probe"))
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--terrain-level", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for case, hipy, knee in (("baseline", 0., 0.), ("hipy", -.1, 0.),
                             ("knee", 0., -.1), ("both", -.1, -.1)):
        output = args.output_dir / f"{case}.json"
        print(f"Running {case}: HipY {hipy:+.2f}, Knee {knee:+.2f} rad", flush=True)
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("eval_locomotion.py")),
            "--checkpoint", str(args.checkpoint), "--stage", "terrain",
            "--terrain", "stairs_inv", "--terrain-level", str(args.terrain_level),
            "--vx", str(args.vx), "--num-envs", str(args.num_envs),
            "--seconds", str(args.seconds), "--seed", str(args.seed),
            "--probe-fr-hipy", str(hipy), "--probe-fr-knee", str(knee),
            "--output", str(output),
        ], check=True)
        data = json.loads(output.read_text())
        summary.append({
            "case": case, "falls": data["falls"],
            "forward_vx": data["mean_body_forward_vx_mps"],
            "linear_rmse": data["linear_rmse_mps"], "yaw_rmse": data["yaw_rmse_radps"],
            "stall_fraction": data["forward_command_stall_fraction"],
            "fr_shank_only_fraction": data["shank_without_foot_fraction"][0],
            "fr_clean_touchdown_fraction": data["touchdown_without_prior_shank_only_fraction"][0],
            "fr_duty_factor": data["duty_factor"][0],
            "action_probe": data["action_probe"],
        })
    result = {"checkpoint": str(args.checkpoint), "terrain_level": args.terrain_level,
              "vx": args.vx, "num_envs": args.num_envs, "seconds": args.seconds,
              "seed": args.seed, "cases": summary}
    dest = args.output_dir / "comparison.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    main()
