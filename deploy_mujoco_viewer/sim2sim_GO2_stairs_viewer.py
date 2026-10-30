from legged_gym.envs.GO2_Trot.GO2_Stairs_config import GO2_Stairs_Cfg_Yu
import math
import numpy as np
import mujoco, mujoco_viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import  Logger
import torch
import keyboard  # 导入keyboard库
from pynput import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

# 设置matplotlib全局字体和样式
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 18,  # 坐标轴标签18号
    'axes.titlesize': 20,
    'xtick.labelsize': 20,  # 刻度标签20号
    'ytick.labelsize': 20,  # 刻度标签20号
    'legend.fontsize': 15,
    'figure.titlesize': 20,
    'lines.linewidth': 2.0,
    'lines.markersize': 6
})

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = True
joystick_opened = False

# 添加数据存储的全局变量
data_log = {
    'time': [],
    'joint_angles': [],  # 第一条腿的关节角度
    'joint_velocities': [],  # 第一条腿的关节角速度
    'euler_angles': []  # 欧拉角
}
start_logging = False

def on_press(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    try:
        if key.char == '6':
            x_vel_cmd += 0.1
        elif key.char == '7':
            x_vel_cmd -= 0.3
        elif key.char == '8':
            y_vel_cmd += 0.3
        elif key.char == '9':
            y_vel_cmd -= 0.3
        elif key.char == '-':
            yaw_vel_cmd += 0.5
        elif key.char == '=':
            yaw_vel_cmd -= 0.5
        elif key.char == '1':
            x_vel_cmd=0
            y_vel_cmd=0
            yaw_vel_cmd=0
        print(f"Updated velocities: vx={x_vel_cmd}, vy={y_vel_cmd}, dyaw={yaw_vel_cmd}")
    except AttributeError:
        pass
    
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    # print(data.qpos.astype(np.double).shape,data.qvel.astype(np.double).shape)
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    foot_positions = []
    # foot_forces = data.cfrc_ext[0][2].copy().astype(np.double)
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        # print(body_name)
        if 'foot' in body_name: 
            # print(body_name)
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp + (target_dq - dq)* kd
    return torque_out

def format_euler_angle(value, pos):
    """格式化欧拉角，最多保留小数点后两位"""
    if abs(value) < 1e-6:  # 接近零的值
        return '0.00'
    else:
        # 保留两位小数
        return f'{value:.2f}'

def setup_plots():
    """设置matplotlib图形"""
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 关节角度图
    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Joint Angles (rad)', fontsize=18)
    ax1.set_title('First Leg Joint Angles (3 joints)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    lines1 = []
    for i in range(3):
        line, = ax1.plot([], [], label=f'Joint {i+1}', linewidth=2)
        lines1.append(line)
    
    # 图例放在右上角
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=15)
    
    # 关节角速度图
    ax2.set_xlabel('Time (s)', fontsize=18)
    ax2.set_ylabel('Joint Velocities (rad/s)', fontsize=18)
    ax2.set_title('First Leg Joint Velocities (3 joints)', fontsize=20, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    
    lines2 = []
    for i in range(3):
        line, = ax2.plot([], [], label=f'Joint {i+1}', linewidth=2)
        lines2.append(line)
    
    # 图例放在右上角
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=15)
    
    # 欧拉角图
    ax3.set_xlabel('Time (s)', fontsize=18)
    ax3.set_ylabel('Euler Angles (rad)', fontsize=18)
    ax3.set_title('Base Euler Angles (Roll, Pitch, Yaw)', fontsize=20, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    
    # 设置Y轴刻度格式为最多两位小数
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_euler_angle))
    
    lines3 = []
    labels = ['Roll', 'Pitch', 'Yaw']
    for i, label in enumerate(labels):
        line, = ax3.plot([], [], label=label, linewidth=2)
        lines3.append(line)
    
    # 图例放在右上角
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=15)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3), (lines1, lines2, lines3)

def update_plots(fig, axes, lines):
    """更新图形"""
    global data_log
    
    if len(data_log['time']) > 0:
        time_data = np.array(data_log['time'])
        joint_angles = np.array(data_log['joint_angles'])
        joint_velocities = np.array(data_log['joint_velocities'])
        euler_angles = np.array(data_log['euler_angles'])
        
        # 更新关节角度图
        for i in range(3):
            lines[0][i].set_data(time_data, joint_angles[:, i])
        
        # 更新关节角速度图
        for i in range(3):
            lines[1][i].set_data(time_data, joint_velocities[:, i])
        
        # 更新欧拉角图
        for i in range(3):
            lines[2][i].set_data(time_data, euler_angles[:, i])
        
        # 调整坐标轴范围
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        fig.canvas.draw()
        plt.pause(0.01)

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    global start_logging, data_log
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos

    mujoco.mj_step(model, data)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
   
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(6):
        hist_obs.append(np.zeros([1, 45], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    
    stop_state_log = 4000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    policy_input = np.zeros([1, cfg.env.num_observations*6], dtype=np.float32)
    
    # 设置matplotlib图形
    fig, axes, lines = setup_plots()
    
    for sim_step in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
        
        # 在400次更新后开始记录数据
        if count_lowlevel >= 400 and not start_logging:
            start_logging = True
            print("Started logging data after 400 updates")
            data_log = {key: [] for key in data_log.keys()}  # 清空之前的数据
        
        # 记录数据
        if start_logging:
            # 第一条腿的关节索引 (假设前3个关节是第一条腿)
            leg1_indices = slice(0, 3)  # 第一条腿的3个关节
            
            data_log['time'].append(sim_step * cfg.sim_config.dt)
            data_log['joint_angles'].append(q[leg1_indices].copy())
            data_log['joint_velocities'].append(dq[leg1_indices].copy())
            
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            data_log['euler_angles'].append(eu_ang.copy())
            
            # 每10步更新一次图形
            if len(data_log['time']) % 10 == 0:
                update_plots(fig, axes, lines)
        
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, 45], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 1] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 2] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
            obs[0, 3:6] =omega*cfg.normalization.obs_scales.ang_vel 
            obs[0, 6:9] = gvec

            obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 33:45] = action
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()
            
            for i in range(6):
                policy_input[0, i * 45 : (i + 1) * 45] = hist_obs[i][0, :]
            # 如果 policy_input 在 GPU，先移到 CPU 并转成 numpy

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # Generate PD control
        if sim_step < 100:
            tau = pd_control(np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        else:
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
        data.ctrl = tau
        applied_tau = data.actuator_force

        mujoco.mj_step(model, data)

        viewer.render()
        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + cfg.robot_config.default_dof_pos

    viewer.close()
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图形

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default=f'{LEGGED_GYM_ROOT_DIR}/logs/go2_stairs_raw/exported/policies/policy_dwaq.pt',help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(GO2_Stairs_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml'
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array( [0.,0.8,-1.5,
                -0.0,0.8,-1.5,
                 0.0,0.8,-1.5,
                -0.0,0.8 ,-1.5], dtype=np.double)


    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())