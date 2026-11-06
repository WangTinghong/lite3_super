"""Compare the policy's interpolated terrain to direct static-geometry rays.

This observes geometry without modifying the height map used by the policy.
It audits sparse foot-path samples, not complete mesh coverage.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import feet_world
from lite3_supermimic.target import terrain as tr


class TerrainAudit:
    def __init__(self, env):
        import mujoco
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        exclude = {int(g) for g in env.scene["robot"].indexing.geom_ids}
        self.geoms = [g for g in range(self.model.ngeom) if g not in exclude]
        self.errors, self.near_errors = [], []
        self.edge_count = self.missed_edges = self.no_hits = 0

    def update(self, state, pose, field):
        if field is None:
            return
        import mujoco
        feet = feet_world(state, pose, field)[..., :2]
        heading = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
        offsets = state.new_tensor([0.0, 0.05, 0.12, 0.25])
        xy = feet[:, :, None] + heading[:, None, None] * offsets[None, None, :, None]
        interpolated = tr.ground(xy, field).cpu().numpy()
        points = xy.cpu().numpy().reshape(-1, 2)
        exact = np.full(len(points), np.nan)
        saved = self.model.geom_group.copy()
        try:
            self.model.geom_group[:] = 5
            self.model.geom_group[self.geoms] = 0
            group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            gid = np.zeros(1, dtype=np.int32)
            for i, (x, y) in enumerate(points):
                d = mujoco.mj_ray(self.model, self.data, np.array([x, y, 20.0]),
                                  np.array([0.0, 0.0, -1.0]), group, 1, -1, gid)
                if d >= 0:
                    exact[i] = 20.0 - d
        finally:
            self.model.geom_group[:] = saved
        exact = exact.reshape(interpolated.shape)
        finite = np.isfinite(exact)
        self.no_hits += int((~finite).sum())
        error = np.abs(exact - interpolated)
        self.errors.extend(error[finite].tolist())
        edge = finite.all(-1) & (np.ptp(exact, axis=-1) > 0.04)
        self.edge_count += int(edge.sum())
        self.missed_edges += int((edge & (np.ptp(interpolated, axis=-1) <= 0.04)).sum())
        self.near_errors.extend(error[edge].reshape(-1).tolist())

    def result(self):
        def stats(values):
            if not values:
                return {"samples": 0}
            return dict(samples=len(values), mae_m=float(np.mean(values)),
                        p95_abs_error_m=float(np.quantile(values, .95)),
                        max_abs_error_m=float(np.max(values)))
        return dict(all=stats(self.errors), edge_paths=stats(self.near_errors),
                    exact_edge_paths=self.edge_count,
                    missed_edge_fraction=self.missed_edges / self.edge_count if self.edge_count else None,
                    no_ray_hits=self.no_hits,
                    note="Sparse vertical rays at feet and 5/12/25 cm ahead; no contact-force approximation.")
