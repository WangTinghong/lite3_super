"""Evaluate velocity tracking/gait, record MP4, or open the native MuJoCo viewer."""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from lite3_supermimic.locomotion.config import STAGES
from lite3_supermimic.locomotion.core import foot_clearances, validate_checkpoint
from lite3_supermimic.locomotion.metrics import Metrics, read_contacts, read_shank_contacts
from lite3_supermimic.robot.constants import JOINT_NAMES
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.train import Actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0", help="CPU is for small diagnostics only")
    parser.add_argument("--stage", choices=tuple(STAGES), help="Default: saved stage")
    parser.add_argument("--terrain", help="All envs use this terrain type")
    parser.add_argument("--terrain-level", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--random-commands", action="store_true", help="Exercise stage command changes")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video")
    parser.add_argument("--viewer", action="store_true", help="Native MuJoCo window; requires DISPLAY on Linux")
    parser.add_argument("--keyboard", action="store_true", help="Change velocity commands with keys in --viewer")
    parser.add_argument("--output", help="JSON metrics file")
    parser.add_argument("--gait-csv", help="Env 0 contact/clearance/velocity timeline")
    parser.add_argument("--joint-csv", help="Env 0 joint position/target/velocity/actuator torque timeline")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.seconds <= 0:
        parser.error("num-envs and seconds must be positive")
    if args.viewer and args.video:
        parser.error("Choose --viewer or --video for one run")
    if args.keyboard and not args.viewer:
        parser.error("--keyboard requires --viewer")
    if args.keyboard and args.random_commands:
        parser.error("--keyboard cannot be combined with --random-commands")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("Evaluation requires NVIDIA CUDA")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(ckpt)
    from lite3_supermimic.locomotion.env import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.viewer_config import ViewerConfig
    stage = args.stage or ckpt["cfg"]["stage"]
    harder_stairs = bool(ckpt["cfg"].get("harder_stairs", False)) and stage == "terrain"
    cfg = make_env_cfg(stage, play=True, harder_stairs=harder_stairs)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1 if args.viewer else args.num_envs
    cfg.auto_reset = args.viewer
    # Manual resets preserve terminal kinematics for quantitative evaluation.
    cmd_cfg = cfg.commands["velocity"]
    cmd_cfg.fixed_command = None if args.random_commands else ((0.0, 0.0, 0.0) if args.keyboard
                                                              else (args.vx, args.vy, args.yaw_rate))
    terrain = cfg.scene.terrain.terrain_generator
    if terrain:
        if not 0 <= args.terrain_level < terrain.num_rows:
            parser.error(f"terrain-level must be 0..{terrain.num_rows - 1}")
        cmd_cfg.fixed_level = args.terrain_level
        if args.terrain:
            if args.terrain not in terrain.sub_terrains:
                parser.error(f"terrain must be one of {tuple(terrain.sub_terrains)}")
            # Generate only this type. Column index is not assumed to equal dict order.
            tile = terrain.sub_terrains[args.terrain]
            tile.proportion = 1.0
            terrain.sub_terrains = {args.terrain: tile}
    elif args.terrain or args.terrain_level != 0:
        parser.error("terrain selection requires --stage terrain or adaptive")
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "robot"
    cfg.viewer.distance, cfg.viewer.elevation = 3.2, -20.0
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array" if args.video else None)
    writer = joint_writer = video = None
    try:
        env.common_step_counter = cmd_cfg.curriculum_steps
        obs, _ = env.reset()
        actor = Actor(obs["actor"].shape[-1], tuple(ckpt["cfg"]["actor_hidden"]), 12).to(env.device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        if args.viewer:
            from mjlab.viewer import NativeMujocoViewer
            if args.keyboard:
                from lite3_supermimic.locomotion.teleop import KeyboardVelocityControl
                control = KeyboardVelocityControl(STAGES[stage])
                term = env.command_manager.get_term("velocity")

                def policy(ob):
                    command = control.snapshot()
                    term.cfg.fixed_command = command  # Keep the command after resets/resampling.
                    term.command[:] = term.command.new_tensor(command)
                    return actor(ob["actor"])

                print("MuJoCo 窗口获得键盘焦点后：W/S 前进/后退，J/L 左/右横移，"
                      "Q/E 左/右转，X 停止。每按一次改变速度，速度上限由训练阶段决定。", flush=True)
                viewer = NativeMujocoViewer(env, policy, key_callback=control.on_key)
            else:
                viewer = NativeMujocoViewer(env, lambda ob: actor(ob["actor"]))
            with torch.inference_mode():
                viewer.run(num_steps=max(1, int(args.seconds / env.step_dt)))
            return
        if args.video:
            import mediapy
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            video = mediapy.VideoWriter(args.video, shape=(720, 1280), fps=round(1 / env.step_dt))
            video.__enter__()
        csvwriter = None
        if args.gait_csv:
            Path(args.gait_csv).parent.mkdir(parents=True, exist_ok=True)
            writer = open(args.gait_csv, "w", newline="")
            csvwriter = csv.writer(writer)
            csvwriter.writerow(["time_s", "cmd_vx", "cmd_vy", "cmd_yaw", "vx", "vy", "yaw_rate",
                                "terminated", "truncated"] + [f"contact_{leg}" for leg in st.LEGS]
                               + [f"clearance_{leg}_m" for leg in st.LEGS]
                               + [f"shank_contact_{leg}" for leg in st.LEGS])
        joint_csvwriter = None
        if args.joint_csv:
            robot = env.scene["robot"]
            if tuple(robot.joint_names) != JOINT_NAMES:
                raise ValueError("Unexpected Lite3 joint order in joint CSV")
            Path(args.joint_csv).parent.mkdir(parents=True, exist_ok=True)
            joint_writer = open(args.joint_csv, "w", newline="")
            joint_csvwriter = csv.writer(joint_writer)
            joint_csvwriter.writerow(["time_s", "cmd_vx", "vx", "base_height_m", "base_vz_mps",
                                      "terminated", "truncated"] +
                                     [f"{name}_pos_rad" for name in JOINT_NAMES] +
                                     [f"{name}_target_rad" for name in JOINT_NAMES] +
                                     [f"{name}_vel_radps" for name in JOINT_NAMES] +
                                     [f"{name}_actuator_Nm" for name in JOINT_NAMES] +
                                     [f"foot_lateral_{leg}_m" for leg in st.LEGS])
        term = env.command_manager.get_term("velocity")
        metrics = Metrics(env.num_envs, env.device)
        prev = torch.zeros(env.num_envs, 12, device=env.device)
        steps = max(1, int(args.seconds / env.step_dt))
        with torch.inference_mode():
            for i in range(steps):
                s0, p0 = st.robot_state(env, term.height_field), st.planar_pose(env)
                cmd = term.command.clone()
                a = actor(obs["actor"])
                obs, _, terminated, truncated, _ = env.step(a)
                s1, p1 = st.robot_state(env, term.height_field), st.planar_pose(env)
                contact, force = read_contacts(env.scene["feet_ground_contact"])
                shank_contact, shank_force = read_shank_contacts(env.scene["shanks_ground_contact"])
                metrics.update(s0, p0, s1, p1, cmd, a, prev, contact, force,
                               terminated, truncated, env.step_dt, term.height_field,
                               shank_contact, shank_force)
                data = env.scene["robot"].data
                metrics.update_joint(data.joint_pos, data.joint_pos_target)
                if csvwriter:
                    clear = foot_clearances(s1, p1, term.height_field)[0].tolist()
                    csvwriter.writerow([(i + 1) * env.step_dt] + cmd[0].tolist()
                                       + s1[0, [4, 5, 9]].tolist()
                                       + [bool(terminated[0]), bool(truncated[0])]
                                       + contact[0].int().tolist() + clear
                                       + shank_contact[0].int().tolist())
                if joint_csvwriter:
                    joint_csvwriter.writerow([(i + 1) * env.step_dt, float(cmd[0, 0]),
                                              float(s1[0, 4]), float(s1[0, 0]), float(s1[0, 6]),
                                              bool(terminated[0]), bool(truncated[0])]
                                             + data.joint_pos[0].tolist()
                                             + data.joint_pos_target[0].tolist()
                                             + data.joint_vel[0].tolist()
                                             + data.qfrc_actuator[0].tolist()
                                             + s1[0, st.FEET].reshape(4, 3)[:, 1].tolist())
                if video:
                    video.add_image(env.render())
                done = terminated | truncated
                prev = torch.where(done[:, None], torch.zeros_like(a), a)
                if done.any():
                    obs, _ = env.reset(env_ids=done.nonzero().flatten())
        result = {"checkpoint": args.checkpoint, "stage": stage, "terrain": args.terrain,
                  "terrain_level": args.terrain_level, "command": cmd_cfg.fixed_command,
                  "random_commands": args.random_commands, "seed": args.seed,
                  "num_envs": env.num_envs, **metrics.result()}
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if writer:
            writer.close()
        if joint_writer:
            joint_writer.close()
        if video:
            video.__exit__(None, None, None)
        env.close()


if __name__ == "__main__":
    main()
