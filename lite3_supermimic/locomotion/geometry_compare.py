"""Read-only bilinear-grid reconstruction from a supplied exact height oracle."""
import numpy as np


def ray_grid_interpolate(xy, origin, resolution, ray_heights):
    xy = np.asarray(xy, dtype=np.float64)
    grid = (xy - np.asarray(origin)) / resolution
    cell = np.floor(grid)
    fraction = grid - cell
    offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    corners = (cell[:, None, :] + offsets) * resolution + np.asarray(origin)
    unique, inverse = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
    heights = ray_heights(unique)[inverse].reshape(-1, 4)
    fx, fy = fraction.T
    weights = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), -1)
    return (heights * weights).sum(-1), corners, heights


def error_stats(prediction, exact):
    signed = np.asarray(prediction) - np.asarray(exact)
    valid = np.isfinite(signed)
    v = signed[valid]
    if not len(v):
        return {"samples": 0, "invalid": int(len(signed))}
    return dict(samples=len(v), invalid=int((~valid).sum()),
                mae_m=float(np.abs(v).mean()), bias_m=float(v.mean()),
                p95_abs_error_m=float(np.quantile(np.abs(v), .95)),
                max_abs_error_m=float(np.abs(v).max()),
                underestimate_over_5cm_fraction=float((v < -.05).mean()),
                overestimate_over_5cm_fraction=float((v > .05).mean()))
