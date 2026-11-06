"""Reproducible phase-one collection and offline fitting; no policy training."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child_environment():
    # New server terminals previously lost this setting. Only add the builtins
    # matching the installed Torch CUDA version; never mix arbitrary versions.
    import torch
    env = os.environ.copy()
    version = torch.version.cuda
    if version:
        root = Path(sysconfig.get_paths()["purelib"])
        libraries = sorted(root.glob(f"nvidia/**/libnvrtc-builtins.so.{version}*"))
        if libraries:
            directory = str(libraries[0].parent)
            env["LD_LIBRARY_PATH"] = directory + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--mode", choices=("smoke", "collect", "fit", "holdout"), required=True)
    p.add_argument("--root", type=Path, default=Path("audit/contact_phase1"))
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--seed", type=int, help="Required for holdout collection; use an unused seed")
    args = p.parse_args()
    if args.mode == "holdout" and (args.seed is None or args.seed < 3):
        p.error("holdout requires --seed >= 3 (original splits use 0, 1, 2)")
    if args.mode != "holdout" and args.seed is not None:
        p.error("--seed is only supported for holdout")
    source = args.checkpoint.expanduser().resolve()
    if not source.is_file():
        p.error(f"Checkpoint missing: {source}")
    if args.updates < 1:
        p.error("updates must be positive")
    from lite3_supermimic.locomotion.contact_data import file_hash
    source_hash = file_hash(source)
    env = child_environment()
    root = args.root.resolve()
    smoke = args.mode == "smoke"
    if smoke:
        root = root / "smoke"
    scenes = [("up5_v05", "terrain", "stairs_inv", 5, .5)] if smoke else [
        ("flat_v05", "flat", None, 0, .5),
        ("up5_v05", "terrain", "stairs_inv", 5, .5),
        ("up8_v10", "terrain", "stairs_inv", 8, 1.0),
        ("down5_v05", "terrain", "stairs", 5, .5),
    ]
    splits = [("holdout", args.seed)] if args.mode == "holdout" else [
        ("train", 0), ("validation", 1), ("test", 2)]
    paths = {name: [] for name, _ in splits}
    for split, seed in splits:
        for name, stage, terrain, level, speed in scenes:
            directory = root / "data" / split / name
            paths[split].append(str(directory))
            if args.mode == "fit":
                continue
            manifest = directory / "manifest.json"
            seconds, num_envs = (3, 16) if smoke else (20, 64)
            if manifest.exists():
                meta = json.loads(manifest.read_text())["metadata"]
                expected = dict(checkpoint_sha256=source_hash, seed=seed, stage=stage,
                                terrain=terrain, terrain_level=level, command=[speed, 0.0, 0.0],
                                seconds=seconds, num_envs=num_envs)
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Existing collection has different settings: {directory}")
                print(f"Already complete: {directory}", flush=True)
                continue
            command = [sys.executable, "-u", str(ROOT / "scripts/eval_locomotion.py"),
                       "--checkpoint", str(source), "--stage", stage, "--terrain-level", str(level),
                       "--vx", str(speed), "--seed", str(seed), "--num-envs", str(num_envs),
                       "--seconds", str(seconds), "--contact-data", str(directory),
                       "--output", str(root / "metrics" / f"{split}_{name}.json")]
            if terrain:
                command += ["--terrain", terrain]
            print(f"Collecting {split}/{name} (policy frozen)", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.mode in ("smoke", "fit"):
        command = [sys.executable, "-u", str(ROOT / "scripts/fit_contact_world_model.py"),
                   "--checkpoint", str(source), "--output-dir", str(root / "fit"),
                   "--updates", str(10 if smoke else args.updates)]
        for key, values in paths.items():
            command += ["--" + key, *values]
        if smoke:
            command += ["--batch", "32", "--eval-windows", "128", "--eval-interval", "10"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"Completed {args.mode}: {root}", flush=True)


if __name__ == "__main__":
    main()
