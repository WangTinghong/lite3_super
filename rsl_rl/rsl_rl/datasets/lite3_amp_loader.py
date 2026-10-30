import json
import os
from typing import Iterable

import torch


class Lite3AMPLoader:
    """Validated AMP transitions for the Lite3 motion data format."""

    FRAME_DIM = 61
    OBSERVATION_DIM = 30

    ROOT_ROT = slice(3, 7)
    JOINT_POS = slice(7, 19)
    LINEAR_VEL = slice(31, 34)
    ANGULAR_VEL = slice(34, 37)
    JOINT_VEL = slice(37, 49)

    def __init__(self, motion_files: Iterable[str], transition_dt: float, device: str):
        self.device = torch.device(device)
        self.transition_dt = float(transition_dt)

        states = []
        next_states = []
        transition_weights = []
        self.motion_names = []

        files = list(motion_files)
        if not files:
            raise ValueError("At least one Lite3 AMP motion file is required")

        for motion_file in files:
            motion_file = os.path.abspath(os.path.expanduser(motion_file))
            with open(motion_file, "r", encoding="utf-8") as stream:
                motion = json.load(stream)

            frames = torch.as_tensor(motion["Frames"], dtype=torch.float32)
            self._validate_frames(frames, motion_file)

            frame_dt = float(motion["FrameDuration"])
            stride_float = self.transition_dt / frame_dt
            stride = int(round(stride_float))
            if stride < 1 or abs(stride_float - stride) > 1e-5:
                raise ValueError(
                    f"{motion_file}: transition_dt={self.transition_dt} must be an integer "
                    f"multiple of FrameDuration={frame_dt}"
                )
            if frames.shape[0] <= stride:
                raise ValueError(f"{motion_file}: not enough frames for transition stride {stride}")

            amp_frames = self._to_amp_observations(frames, motion_file)
            motion_states = amp_frames[:-stride]
            motion_next_states = amp_frames[stride:]
            motion_weight = float(motion.get("MotionWeight", 1.0))
            if motion_weight <= 0.0:
                raise ValueError(f"{motion_file}: MotionWeight must be positive")

            states.append(motion_states)
            next_states.append(motion_next_states)
            transition_weights.append(
                torch.full(
                    (motion_states.shape[0],),
                    motion_weight / motion_states.shape[0],
                    dtype=torch.float32,
                )
            )
            self.motion_names.append(os.path.basename(motion_file))

        self.states = torch.cat(states, dim=0).to(self.device)
        self.next_states = torch.cat(next_states, dim=0).to(self.device)
        self.sample_weights = torch.cat(transition_weights, dim=0).to(self.device)
        self.sample_weights /= self.sample_weights.sum()

    @classmethod
    def _validate_frames(cls, frames: torch.Tensor, motion_file: str) -> None:
        if frames.ndim != 2 or frames.shape[1] != cls.FRAME_DIM:
            raise ValueError(
                f"{motion_file}: expected [num_frames, {cls.FRAME_DIM}], got {tuple(frames.shape)}"
            )
        if not torch.isfinite(frames).all():
            raise ValueError(f"{motion_file}: motion contains non-finite values")

    @classmethod
    def _to_amp_observations(cls, frames: torch.Tensor, motion_file: str) -> torch.Tensor:
        # The files store xyzw root rotations and world-frame base velocities.
        root_rotation = frames[:, cls.ROOT_ROT]
        quaternion_norm = torch.linalg.vector_norm(root_rotation, dim=-1, keepdim=True)
        if torch.any(quaternion_norm < 1e-6):
            raise ValueError(f"{motion_file}: motion contains a zero-length root quaternion")
        root_rotation = root_rotation / quaternion_norm

        joint_pos = -frames[:, cls.JOINT_POS]
        linear_vel = cls._quat_rotate_inverse(root_rotation, frames[:, cls.LINEAR_VEL])
        angular_vel = cls._quat_rotate_inverse(root_rotation, frames[:, cls.ANGULAR_VEL])
        joint_vel = -frames[:, cls.JOINT_VEL]
        amp_obs = torch.cat((joint_pos, linear_vel, angular_vel, joint_vel), dim=-1)
        if amp_obs.shape[1] != cls.OBSERVATION_DIM:
            raise RuntimeError(f"Unexpected Lite3 AMP observation width: {amp_obs.shape[1]}")
        return amp_obs

    @staticmethod
    def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        quat_vector = quaternion[:, :3]
        quat_scalar = quaternion[:, 3:4]
        return (
            vector * (2.0 * quat_scalar.square() - 1.0)
            - 2.0 * quat_scalar * torch.linalg.cross(quat_vector, vector, dim=-1)
            + 2.0 * quat_vector * torch.sum(quat_vector * vector, dim=-1, keepdim=True)
        )

    def sample(self, batch_size: int):
        indices = torch.multinomial(self.sample_weights, batch_size, replacement=True)
        return self.states[indices], self.next_states[indices]

    @property
    def observation_dim(self) -> int:
        return self.OBSERVATION_DIM

    @property
    def num_transitions(self) -> int:
        return self.states.shape[0]

