import sys
from pathlib import Path
PATH_PARENT = Path(__file__).resolve().parent
DEPLOY_ROOT_DIR = PATH_PARENT.parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

import os
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import imageio
from argparse import ArgumentParser
from collections import deque
import pygame
from matplotlib import pyplot as plt

reset_requested = False

def resolve_project_path(path_value):
    path_value = str(path_value).replace("{DEPLOY_ROOT_DIR}", str(DEPLOY_ROOT_DIR))
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = DEPLOY_ROOT_DIR / path
    return path.resolve()

def require_file(path, description):
    if path.is_file():
        return

    message = f"{description} not found: {path}"
    if description == "MuJoCo XML":
        xml_dir = PATH_PARENT / "resources" / "go2"
        available = sorted(xml_dir.glob("*.xml"))
        if available:
            choices = "\n".join(f"  - {item}" for item in available)
            message += f"\nAvailable MuJoCo XML files:\n{choices}"
    raise FileNotFoundError(message)

def get_mean_vel_motion_mask(mean_vel, current_obs, cmd_threshold, vel_threshold):
    motion_mask = torch.zeros((mean_vel.shape[0], 1), dtype=torch.bool, device=mean_vel.device)
    if cmd_threshold > 0.0:
        cmd_norm = torch.linalg.norm(current_obs[:, :3], dim=1, keepdim=True)
        motion_mask = motion_mask | (cmd_norm > cmd_threshold)
    if vel_threshold > 0.0:
        planar_vel_norm = torch.linalg.norm(mean_vel[:, :2], dim=1, keepdim=True)
        motion_mask = motion_mask | (planar_vel_norm > vel_threshold)
    return motion_mask

def calibrate_mean_vel(
    mean_vel,
    mean_vel_scale=None,
    mean_vel_bias=None,
    motion_mask=None,
    stationary_zero=False,
    stationary_zero_axes=None,
):
    calibrated_mean_vel = mean_vel
    if mean_vel_bias is not None:
        calibrated_mean_vel = calibrated_mean_vel + mean_vel_bias
    if mean_vel_scale is not None:
        calibrated_mean_vel = calibrated_mean_vel * mean_vel_scale
    if motion_mask is not None and stationary_zero_axes is not None:
        axes = stationary_zero_axes.to(device=mean_vel.device, dtype=torch.bool).view(1, -1)
        stationary_mean_vel = torch.where(axes, torch.zeros_like(mean_vel), mean_vel) if stationary_zero else mean_vel
        calibrated_mean_vel = torch.where(motion_mask, calibrated_mean_vel, stationary_mean_vel)
    return calibrated_mean_vel

def run_policy(
    policy,
    obs_tensor,
    num_obs,
    deterministic_vae,
    mean_vel_scale=None,
    mean_vel_bias=None,
    calibration_cmd_threshold=0.0,
    calibration_vel_threshold=0.0,
    stationary_zero=False,
    stationary_zero_axes=None,
):
    if deterministic_vae and hasattr(policy, "vae") and hasattr(policy, "actor"):
        try:
            obs_history = obs_tensor[:, :-num_obs]
            current_obs = obs_tensor[:, -num_obs:]
            _, _, _, latent = policy.vae.cenet_forward(obs_history, True)
            mean_vel, _, mean_latent, _ = latent
            motion_mask = get_mean_vel_motion_mask(
                mean_vel, current_obs, calibration_cmd_threshold, calibration_vel_threshold
            )
            mean_vel = calibrate_mean_vel(
                mean_vel,
                mean_vel_scale,
                mean_vel_bias,
                motion_mask,
                stationary_zero,
                stationary_zero_axes,
            )
            return policy.actor(torch.cat((mean_vel, mean_latent, current_obs), dim=1))
        except Exception:
            pass
    return policy(obs_tensor)

def estimate_policy_vel(policy, obs_tensor):
    if not hasattr(policy, "estimate_vel"):
        raise AttributeError(
            "Loaded policy does not expose estimate_vel(). Re-export policy_dwaq.pt after adding the mean_vel JIT interface."
        )
    return policy.estimate_vel(obs_tensor)

def save_mean_vel_plot(times, mean_vels, sim_vels, path):
    times = np.asarray(times)
    mean_vels = np.asarray(mean_vels)
    sim_vels = np.asarray(sim_vels)
    labels = ("Vx", "Vy", "Vz")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, sim_vels[:, i], label="MuJoCo local lin vel", linewidth=1.4)
        ax.plot(times, mean_vels[:, i], label="VAE mean_vel", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"{labels[i]} (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("VAE mean_vel vs MuJoCo local linear velocity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_base_yaw(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_base_velocities(data):
    # MuJoCo free-joint linear velocity is world-frame, while angular velocity
    # is already expressed in the body's local frame.
    local_linear_velocity = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
    local_angular_velocity = np.asarray(data.qvel[3:6], dtype=np.float32)
    return local_linear_velocity, local_angular_velocity

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def request_reset_from_viewer(keycode):
    global reset_requested
    # MuJoCo viewer uses GLFW key codes: R=82, Backspace=259, Delete=261.
    if keycode in (82, 259, 261):
        reset_requested = True

def reset_mujoco_state(model, data, default_angles, num_actions):
    mujoco.mj_resetData(model, data)
    data.qpos[7 : 7 + num_actions] = default_angles
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

def get_keyboard_command(keys, cmd_limits):
    cmd_x = 0.0
    cmd_y = 0.0
    cmd_yaw = 0.0
    if keys[pygame.K_w]: cmd_x += cmd_limits[0, 1]
    if keys[pygame.K_s]: cmd_x += cmd_limits[0, 0]
    if keys[pygame.K_a]: cmd_y += cmd_limits[1, 1]
    if keys[pygame.K_d]: cmd_y += cmd_limits[1, 0]
    if keys[pygame.K_q]: cmd_yaw += cmd_limits[2, 1]
    if keys[pygame.K_e]: cmd_yaw += cmd_limits[2, 0]
    return np.clip(np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32), cmd_limits[:, 0], cmd_limits[:, 1])

def get_xbox_command(joystick, max_cmd):
    max_cmd = np.array(max_cmd, dtype=np.float32)
    pygame.event.pump()
    dead_zone = 0.1
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone: lx = 0
    if abs(ly) < dead_zone: ly = 0
    if abs(rx) < dead_zone: rx = 0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.clip(np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32), -max_cmd, max_cmd)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PATH_PARENT / "configs" / "go2.yaml"),
        help="Path to the deployment YAML config.",
    )
    parser.add_argument("--policy-path", help="Override policy_path from the config.")
    parser.add_argument("--xml-path", help="Override xml_path from the config.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and load the config, MuJoCo model, and policy without opening windows.",
    )
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    parser.add_argument("--save-moe-latent", action="store_true", help="Whether to save mixture of experts latent vectors.")
    parser.add_argument("--plot-mean-vel", action="store_true", help="Save a plot comparing VAE mean_vel and MuJoCo local linear velocity.")
    parser.add_argument("--mean-vel-plot-path", help="Optional output path for --plot-mean-vel.")
    args = parser.parse_args()
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    save_moe_latent = args.save_moe_latent
    plot_mean_vel = args.plot_mean_vel
    config_path = Path(args.config).expanduser().resolve()
    require_file(config_path, "Config")

    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = resolve_project_path(args.policy_path or config["policy_path"])
        xml_path = resolve_project_path(args.xml_path or config["xml_path"])
        require_file(policy_path, "Policy")
        require_file(xml_path, "MuJoCo XML")

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        torque_limits = np.array(config["torque_limits"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_history_obs = config["num_history_obs"]
        policy_input_dim = config["policy_input_dim"]
        expected_policy_input_dim = (num_history_obs + 1) * num_obs
        if policy_input_dim != expected_policy_input_dim:
            raise ValueError(
                f"policy_input_dim is {policy_input_dim}, expected {expected_policy_input_dim} "
                f"from num_obs={num_obs} and num_history_obs={num_history_obs}."
            )

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        cmd_init = cmd.copy()
        cmd_smoothing = float(config.get("cmd_smoothing", 1.0))
        heading_hold = bool(config.get("heading_hold", True))
        heading_hold_gain = float(config.get("heading_hold_gain", 0.5))
        heading_hold_integral_gain = float(config.get("heading_hold_integral_gain", 0.0))
        heading_hold_integral_limit = float(config.get("heading_hold_integral_limit", 0.2))
        heading_hold_yaw_threshold = float(config.get("heading_hold_yaw_threshold", 0.05))
        yaw_trim = float(config.get("yaw_trim", 0.0))
        line_hold = bool(config.get("line_hold", True))
        line_hold_gain = float(config.get("line_hold_gain", 0.4))
        line_hold_damping = float(config.get("line_hold_damping", 0.2))
        line_hold_limit = float(config.get("line_hold_limit", 0.3))
        line_hold_min_vx = float(config.get("line_hold_min_vx", 0.1))
        line_hold_y_threshold = float(config.get("line_hold_y_threshold", 0.05))
        y_trim = float(config.get("y_trim", 0.0))
        stair_pulse_cmd = bool(config.get("stair_pulse_cmd", False))
        stair_pulse_threshold = float(config.get("stair_pulse_threshold", 0.9))
        stair_pulse_on_time = float(config.get("stair_pulse_on_time", 0.35))
        stair_pulse_off_time = float(config.get("stair_pulse_off_time", 0.22))
        deterministic_vae = bool(config.get("deterministic_vae", True))
        mean_vel_bias = np.array(config.get("mean_vel_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
        if mean_vel_bias.shape != (3,):
            raise ValueError(f"mean_vel_bias must have 3 values, got {mean_vel_bias}.")
        mean_vel_scale = np.array(config.get("mean_vel_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        if mean_vel_scale.shape != (3,):
            raise ValueError(f"mean_vel_scale must have 3 values, got {mean_vel_scale}.")
        mean_vel_calibration_cmd_threshold = float(config.get("mean_vel_calibration_cmd_threshold", 0.0))
        mean_vel_calibration_vel_threshold = float(config.get("mean_vel_calibration_vel_threshold", 0.0))
        mean_vel_stationary_zero = bool(config.get("mean_vel_stationary_zero", False))
        mean_vel_stationary_zero_axes = np.array(
            config.get("mean_vel_stationary_zero_axes", [False, True, False]), dtype=np.bool_
        )
        if mean_vel_stationary_zero_axes.shape != (3,):
            raise ValueError(
                f"mean_vel_stationary_zero_axes must have 3 values, got {mean_vel_stationary_zero_axes}."
            )
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)
        heading_hold_limit = float(config.get("heading_hold_limit", cmd_limits[2, 1]))

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = deque(maxlen=num_history_obs + 1)
    for _ in range(num_history_obs + 1):
        obs_history.append(np.zeros(num_obs, dtype=np.float32))
    policy_input = np.zeros(policy_input_dim, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    if m.nu != num_actions:
        raise ValueError(f"Model has {m.nu} actuators, but config num_actions is {num_actions}.")
    if m.nq < 7 + num_actions or m.nv < 6 + num_actions:
        raise ValueError(
            f"Model state is too small for {num_actions} actions: nq={m.nq}, nv={m.nv}."
        )
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    reset_mujoco_state(m, d, default_angles, num_actions)
    heading_hold_target = get_base_yaw(d.qpos[3:7]) if heading_hold else None
    heading_hold_integral = 0.0
    line_hold_origin = None
    line_hold_heading = None
    
    # load policy
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    mean_vel_bias_tensor = torch.from_numpy(mean_vel_bias).unsqueeze(0)
    mean_vel_scale_tensor = torch.from_numpy(mean_vel_scale).unsqueeze(0)
    mean_vel_stationary_zero_axes_tensor = torch.from_numpy(mean_vel_stationary_zero_axes).unsqueeze(0)
    with torch.no_grad():
        validation_input = torch.zeros(1, policy_input_dim)
        validation_result = run_policy(
            policy,
            validation_input,
            num_obs,
            deterministic_vae,
            mean_vel_scale_tensor,
            mean_vel_bias_tensor,
            mean_vel_calibration_cmd_threshold,
            mean_vel_calibration_vel_threshold,
            mean_vel_stationary_zero,
            mean_vel_stationary_zero_axes_tensor,
        )
        repeated_result = run_policy(
            policy,
            validation_input,
            num_obs,
            deterministic_vae,
            mean_vel_scale_tensor,
            mean_vel_bias_tensor,
            mean_vel_calibration_cmd_threshold,
            mean_vel_calibration_vel_threshold,
            mean_vel_stationary_zero,
            mean_vel_stationary_zero_axes_tensor,
        )
        if plot_mean_vel:
            validation_mean_vel = estimate_policy_vel(policy, validation_input)
            if validation_mean_vel.shape[-1] != 3:
                raise ValueError(
                    f"Policy estimate_vel outputs {validation_mean_vel.shape[-1]} values, expected 3."
                )
    validation_action = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    if validation_action.shape[-1] != num_actions:
        raise ValueError(
            f"Policy outputs {validation_action.shape[-1]} actions, but config num_actions is {num_actions}."
        )
    repeated_action = repeated_result[0] if isinstance(repeated_result, tuple) else repeated_result
    if not torch.equal(validation_action, repeated_action):
        raise ValueError(
            "Policy produces different actions for identical inputs. "
            "Enable deterministic_vae or export the policy with deterministic=True."
        )

    print(f"Config: {config_path}")
    print(f"MuJoCo XML: {xml_path}")
    print(f"Policy: {policy_path}")
    print(f"Deterministic VAE: {deterministic_vae}")
    print(f"Mean velocity bias: {mean_vel_bias.tolist()}")
    print(f"Mean velocity scale: {mean_vel_scale.tolist()}")
    print(f"Mean velocity calibration cmd threshold: {mean_vel_calibration_cmd_threshold}")
    print(f"Mean velocity calibration vel threshold: {mean_vel_calibration_vel_threshold}")
    print(f"Mean velocity stationary zero: {mean_vel_stationary_zero}")
    print(f"Mean velocity stationary zero axes: {mean_vel_stationary_zero_axes.tolist()}")
    if args.check_only:
        print(f"Validation successful: nq={m.nq}, nv={m.nv}, actuators={m.nu}")
        sys.exit(0)

    pygame.display.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((260, 120))
    pygame.display.set_caption("Go2 Keyboard Control")
    print("Keyboard control: W/S=vx, A/D=vy, Q/E=yaw")
    print("Click the pygame keyboard window before pressing keys.")
    use_joystick = False
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    video_fps = 50
    renderer = None
    if save_video:
        renderer = mujoco.Renderer(m, height=360, width=640)
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    if visualize_moe_weights:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.set_title(f"Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")
        bars = None
    
    if save_moe_latent:
        latent_save_dir = str(PATH_PARENT / "data_latents")
        os.makedirs(latent_save_dir, exist_ok=True)
        latent_filename = f"{model_name}_{cmd_str}_latents.npy"
        latent_path = os.path.join(latent_save_dir, latent_filename)
        all_latents = []

    if plot_mean_vel:
        if args.mean_vel_plot_path:
            mean_vel_plot_path = resolve_project_path(args.mean_vel_plot_path)
        else:
            mean_vel_plot_path = PATH_PARENT / "plots" / f"{model_name}_{cmd_str}_mean_vel.png"
        mean_vel_times = []
        mean_vel_estimates = []
        sim_linear_vels = []
        print(f"Mean velocity comparison plot will be saved to: {mean_vel_plot_path}")

    with mujoco.viewer.launch_passive(m, d, key_callback=request_reset_from_viewer) as viewer:

        # set viewer.camera to follow robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            if reset_requested:
                reset_mujoco_state(m, d, default_angles, num_actions)
                action.fill(0.0)
                last_action.fill(0.0)
                target_dof_pos = default_angles.copy()
                obs.fill(0.0)
                policy_input.fill(0.0)
                obs_history.clear()
                for _ in range(num_history_obs + 1):
                    obs_history.append(np.zeros(num_obs, dtype=np.float32))
                cmd = cmd_init.copy()
                heading_hold_target = get_base_yaw(d.qpos[3:7]) if heading_hold else None
                heading_hold_integral = 0.0
                line_hold_origin = None
                line_hold_heading = None
                counter = 0
                reset_requested = False
                print("\nSimulation reset.")

            local_vel, local_ang_vel = get_base_velocities(d)
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
            step_start = time.time()

            if counter % control_decimation == 0:
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DELETE] or keys[pygame.K_BACKSPACE] or keys[pygame.K_r]:
                    reset_requested = True
                    continue
                keyboard_cmd = get_keyboard_command(keys, cmd_limits)
                if np.any(keyboard_cmd):
                    cmd_target = keyboard_cmd
                elif use_joystick:
                    cmd_target = get_xbox_command(joystick, config["max_cmd"])
                else:
                    cmd_target = cmd_init
                if stair_pulse_cmd and abs(cmd_target[0]) >= stair_pulse_threshold:
                    pulse_period = stair_pulse_on_time + stair_pulse_off_time
                    pulse_t = (counter * simulation_dt) % pulse_period
                    if pulse_t > stair_pulse_on_time:
                        cmd_target = cmd_target.copy()
                        cmd_target[0] = 0.0
                manual_yaw = abs(cmd_target[2]) > heading_hold_yaw_threshold
                manual_y = abs(cmd_target[1]) > line_hold_y_threshold
                line_error = 0.0
                line_active = line_hold and abs(cmd_target[0]) > line_hold_min_vx and not manual_y and not manual_yaw
                if line_active:
                    if line_hold_origin is None:
                        line_hold_origin = d.qpos[:2].copy()
                        line_hold_heading = (
                            heading_hold_target if heading_hold_target is not None else get_base_yaw(d.qpos[3:7])
                        )
                    left_axis = np.array([-np.sin(line_hold_heading), np.cos(line_hold_heading)], dtype=np.float32)
                    line_error = float(np.dot(d.qpos[:2] - line_hold_origin, left_axis))
                    y_cmd = -line_hold_gain * line_error - line_hold_damping * local_vel[1] + y_trim
                    cmd_target = cmd_target.copy()
                    cmd_target[1] = np.clip(
                        y_cmd,
                        max(cmd_limits[1, 0], -line_hold_limit),
                        min(cmd_limits[1, 1], line_hold_limit),
                    )
                else:
                    line_hold_origin = None
                    line_hold_heading = None
                heading_error = 0.0
                if heading_hold:
                    current_yaw = get_base_yaw(d.qpos[3:7])
                    heading_active = np.linalg.norm(cmd_target[:2]) > line_hold_min_vx
                    if manual_yaw or not heading_active:
                        heading_hold_target = None
                        heading_hold_integral = 0.0
                    else:
                        if heading_hold_target is None:
                            heading_hold_target = current_yaw
                        heading_error = wrap_to_pi(heading_hold_target - current_yaw)
                        heading_hold_integral = np.clip(
                            heading_hold_integral + heading_error * control_decimation * simulation_dt,
                            -heading_hold_integral_limit,
                            heading_hold_integral_limit,
                        )
                        yaw_cmd = (
                            heading_hold_gain * heading_error
                            + heading_hold_integral_gain * heading_hold_integral
                            + yaw_trim
                        )
                        cmd_target = cmd_target.copy()
                        cmd_target[2] = np.clip(
                            yaw_cmd,
                            max(cmd_limits[2, 0], -heading_hold_limit),
                            min(cmd_limits[2, 1], heading_hold_limit),
                        )
                elif not manual_yaw and yaw_trim != 0.0:
                    cmd_target = cmd_target.copy()
                    cmd_target[2] = np.clip(cmd_target[2] + yaw_trim, cmd_limits[2, 0], cmd_limits[2, 1])
                cmd = (1.0 - cmd_smoothing) * cmd + cmd_smoothing * cmd_target
                show_str += (
                    f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}, "
                    f"HeadErr={heading_error:.3f}, LineErr={line_error:.3f}"
                )
                print(show_str, end='\r')

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            tau = np.clip(tau, -torque_limits, torque_limits)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                _, ang_vel = get_base_velocities(d)

                qj = (qj - default_angles) * dof_pos_scale

                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                scaled_ang_vel = ang_vel * ang_vel_scale

                obs[:3] = cmd * cmd_scale
                obs[3:6] = scaled_ang_vel
                obs[6:9] = gravity_orientation
                obs[9 : 9 + num_actions] = qj[idx_mj2model]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[idx_mj2model]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[idx_mj2model]
                obs_history.append(obs.copy())
                policy_input[:] = np.concatenate(list(obs_history), axis=0)
                obs_tensor = torch.from_numpy(policy_input).unsqueeze(0)
                # policy inference
                last_action = action
                with torch.no_grad():
                    result = run_policy(
                        policy,
                        obs_tensor,
                        num_obs,
                        deterministic_vae,
                        mean_vel_scale_tensor,
                        mean_vel_bias_tensor,
                        mean_vel_calibration_cmd_threshold,
                        mean_vel_calibration_vel_threshold,
                        mean_vel_stationary_zero,
                        mean_vel_stationary_zero_axes_tensor,
                    )
                    if plot_mean_vel:
                        raw_mean_vel = estimate_policy_vel(policy, obs_tensor)
                        motion_mask = get_mean_vel_motion_mask(
                            raw_mean_vel,
                            obs_tensor[:, -num_obs:],
                            mean_vel_calibration_cmd_threshold,
                            mean_vel_calibration_vel_threshold,
                        )
                        mean_vel = calibrate_mean_vel(
                            raw_mean_vel,
                            mean_vel_scale_tensor,
                            mean_vel_bias_tensor,
                            motion_mask,
                            mean_vel_stationary_zero,
                            mean_vel_stationary_zero_axes_tensor,
                        )
                if isinstance(result, tuple):
                    action = result[0]
                    weights = None
                    latent = None
                    if len(result) > 1 and isinstance(result[1], tuple):
                        weights, latent = result[1]  # moe

                    action = action.detach().cpu().numpy().squeeze()[idx_model2mj]

                    if weights is not None:
                        weights = weights.detach().cpu().numpy().squeeze()
                    if latent is not None:
                        latent = latent.detach().cpu().numpy().squeeze()

                    if visualize_moe_weights and weights is not None:
                        if bars is None:
                            x = np.arange(len(weights))
                            bars = ax.bar(x, weights)
                            ax.set_ylim(0, 1)
                        else:
                            for bar, w in zip(bars, weights):
                                bar.set_height(w)
                        
                        plt.draw()
                        plt.pause(0.001) # 这会造成大约 1ms 的延迟
                    if save_moe_latent and latent is not None:
                        all_latents.append(latent)
                else:
                    action = result.detach().cpu().numpy().squeeze()[idx_model2mj]
                if plot_mean_vel:
                    sim_linear_vel, _ = get_base_velocities(d)
                    mean_vel_times.append(counter * simulation_dt)
                    mean_vel_estimates.append(mean_vel.detach().cpu().numpy().squeeze().copy())
                    sim_linear_vels.append(sim_linear_vel.copy())
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # writer.close()
    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()
        renderer.close()
    if save_moe_latent and len(all_latents) > 0:
        all_latents = np.array(all_latents)
        np.save(latent_path, all_latents)
        print(f"Latent vectors saved successfully to {latent_path}")
    if plot_mean_vel:
        if len(mean_vel_times) > 0:
            save_mean_vel_plot(mean_vel_times, mean_vel_estimates, sim_linear_vels, mean_vel_plot_path)
            print(f"Mean velocity comparison plot saved to {mean_vel_plot_path}")
        else:
            print("No mean velocity samples were recorded.")
    pygame.quit()
