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


def common_support_comparison(predictions, exact, edge):
    common = np.isfinite(exact)
    for values in predictions.values():
        common &= np.isfinite(values)
    return dict(total_points=len(exact), common_valid_points=int(common.sum()),
                excluded_points=int((~common).sum()), common_edge_points=int((common & edge).sum()),
                methods={name: {'all': error_stats(v[common], exact[common]),
                                'edge_paths': error_stats(v[common & edge], exact[common & edge])}
                         for name, v in predictions.items()})


def cached_node_audit(field, corners, ray_heights, epsilon=1e-6):
    """Compare unique nodes using original arange coordinates and edge perturbations.

    Missing rays remain missing except in the explicitly labelled builder-compatible
    comparison, which reproduces build_height_field's zero fallback.
    """
    indices = np.rint((corners.reshape(-1, 2) - np.asarray(field.origin)) / field.res).astype(int)
    indices = np.unique(indices, axis=0)
    valid = ((indices[:, 0] >= 0) & (indices[:, 0] < field.h.shape[1])
             & (indices[:, 1] >= 0) & (indices[:, 1] < field.h.shape[0]))
    indices = indices[valid]
    xs = np.arange(field.origin[0], field.origin[0] + field.h.shape[1] * field.res + 1e-9, field.res)
    ys = np.arange(field.origin[1], field.origin[1] + field.h.shape[0] * field.res + 1e-9, field.res)
    original_xy = np.stack((xs[indices[:, 0]], ys[indices[:, 1]]), -1)
    nominal_xy = indices * field.res + np.asarray(field.origin)
    stored = field.h.cpu().numpy()[indices[:, 1], indices[:, 0]]
    original = ray_heights(original_xy)
    nominal = ray_heights(nominal_xy)
    shifts = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]) * epsilon
    neighbors = np.stack([ray_heights(original_xy + shift) for shift in shifts], -1)
    finite = np.isfinite(neighbors).all(-1) & np.isfinite(original)
    span = np.ptp(np.column_stack((original, neighbors)), axis=-1)
    # Count one-sided edge hits as sensitive too, even if the exact node misses.
    sensitive = ~finite | (span > .001)
    stable = ~sensitive
    report = dict(unique_nodes=len(indices), epsilon_m=epsilon,
                  coordinate_generation_max_difference_m=float(np.abs(original_xy - nominal_xy).max()) if len(indices) else None,
                  nominal_no_hits=int((~np.isfinite(nominal)).sum()),
                  original_arange_no_hits=int((~np.isfinite(original)).sum()),
                  edge_sensitive_or_missing_nodes=int(sensitive.sum()), stable_nodes=int(stable.sum()),
                  cached_vs_nominal=error_stats(stored, nominal),
                  cached_vs_original_arange=error_stats(stored, original),
                  cached_vs_original_builder_zero_fallback=error_stats(stored, np.where(np.isfinite(original), original, 0.)),
                  cached_vs_stable_original_nodes=error_stats(stored[stable], original[stable]))
    rows = np.column_stack((indices, original_xy, stored, nominal, original, span, sensitive))
    return report, rows
