"""Collect action-intervention stair data and fit an offline contact world model.

The saved actor is frozen. This script does not train or update a policy.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.contact_phase1 import child_environment
from lite3_supermimic.locomotion.contact_data import file_hash

SPLITS = {"train": (6, 9), "validation": (7,), "test": (8,)}
CASES = {"baseline": (0.0, 0.0), "hipy_minus": (-0.05, 0.0),
         "hipy_plus": (0.05, 0.0), "knee_minus": (0.0, -0.05),
         "knee_plus": (0.0, 0.05)}


def paths(root):
    return {split: [root / "data" / split / f"seed{seed}_{case}"
                    for seed in seeds for case in CASES]
            for split, seeds in SPLITS.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "collect", "fit"), required=True)
    p.add_argument("--root", type=Path, default=Path("audit/contact_action_phase"))
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--initialize-contact-model", type=Path,
                   help="Prior contact model, e.g. phase-one head-refined artifact")
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"Checkpoint missing: {args.checkpoint}")
    if args.initialize_contact_model and not args.initialize_contact_model.is_file():
        p.error(f"Initial contact model missing: {args.initialize_contact_model}")
    if args.num_envs < 1 or args.seconds <= 0 or args.updates < 1:
        p.error("num-envs, seconds and updates must be positive")
    source = args.checkpoint.expanduser().resolve()
    source_hash = file_hash(source)
    root = args.root.resolve() / ("smoke" if args.mode == "smoke" else "")
    num_envs = min(args.num_envs, 4) if args.mode == "smoke" else args.num_envs
    seconds = min(args.seconds, 3.0) if args.mode == "smoke" else args.seconds
    collection = paths(root)
    env = child_environment()
    if args.mode != "fit":
        for split, seeds in SPLITS.items():
            for seed in seeds:
                for case, (hipy, knee) in CASES.items():
                    directory = root / "data" / split / f"seed{seed}_{case}"
                    manifest = directory / "manifest.json"
                    expected = dict(checkpoint_sha256=source_hash, seed=seed,
                                    stage="terrain", terrain="stairs_inv", terrain_level=8,
                                    command=[1.0, 0.0, 0.0], num_envs=num_envs,
                                    seconds=seconds, probe_fr_hipy_rad=hipy,
                                    probe_fr_knee_rad=knee)
                    if manifest.is_file():
                        metadata = json.loads(manifest.read_text())["metadata"]
                        if any(metadata.get(key) != value for key, value in expected.items()):
                            raise ValueError(f"Existing collection settings differ: {directory}")
                        print(f"Already complete: {directory}", flush=True)
                        continue
                    if directory.exists() and any(directory.iterdir()):
                        raise ValueError(f"Incomplete collection at {directory}; inspect it before retrying")
                    command = [sys.executable, "-u", str(ROOT / "scripts/eval_locomotion.py"),
                               "--checkpoint", str(source), "--stage", "terrain",
                               "--terrain", "stairs_inv", "--terrain-level", "8",
                               "--vx", "1.0", "--seed", str(seed),
                               "--num-envs", str(num_envs), "--seconds", str(seconds),
                               "--probe-fr-hipy", str(hipy), "--probe-fr-knee", str(knee),
                               "--contact-data", str(directory),
                               "--output", str(root / "metrics" / f"{split}_seed{seed}_{case}.json")]
                    print(f"Collecting {split} seed {seed} {case}; actor frozen", flush=True)
                    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.mode in ("smoke", "fit"):
        for split, dirs in collection.items():
            for directory in dirs:
                metadata = json.loads((directory / "manifest.json").read_text())["metadata"]
                expected_seed = int(directory.name.split("_")[0][4:])
                expected_case = directory.name.split("_", 1)[1]
                hipy, knee = CASES[expected_case]
                if (metadata.get("checkpoint_sha256") != source_hash
                    or metadata.get("seed") != expected_seed
                    or metadata.get("probe_fr_hipy_rad") != hipy
                    or metadata.get("probe_fr_knee_rad") != knee):
                    raise ValueError(f"Unexpected collection metadata: {directory}")
        command = [sys.executable, "-u", str(ROOT / "scripts/fit_contact_world_model.py"),
                   "--checkpoint", str(source), "--output-dir", str(root / "fit"),
                   "--updates", str(10 if args.mode == "smoke" else args.updates)]
        if args.initialize_contact_model:
            command += ["--initialize-contact-model", str(args.initialize_contact_model.resolve())]
        for split, dirs in collection.items():
            command += ["--" + split, *(str(directory) for directory in dirs)]
        if args.mode == "smoke":
            command += ["--batch", "32", "--eval-windows", "128", "--eval-interval", "10"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"Completed {args.mode}: {root}", flush=True)


if __name__ == "__main__":
    main()
