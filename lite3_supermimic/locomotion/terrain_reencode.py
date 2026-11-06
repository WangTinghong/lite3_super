"""Re-express recorded states in a new terrain reference, preserving physical state."""
import torch
from lite3_supermimic.target import terrain as tr


def reencode_chunk(chunk, old_field, new_field):
    result = dict(chunk)
    for state_key, pose_key in (("s0", "pose0"), ("s1", "pose1")):
        state = chunk[state_key].clone()
        xy = chunk[pose_key][..., :2]
        state[..., 0] += tr.ground(xy, old_field) - tr.ground(xy, new_field)
        result[state_key] = state
    shape = chunk['pose0'].shape
    pose = chunk['pose0'].reshape(-1, 3)
    result['relief'] = tr.relief(pose[:, :2], pose[:, 2], new_field).reshape(*shape[:-1], tr.SCAN_POINTS)
    # All physical velocities, joints, limb positions relative to base, actions,
    # contacts, resets and pose deltas stay unchanged. Keep the original near
    # mask so paired reports describe the same windows rather than new subsets.
    return result


def build_diagnostic_grid(bounds, resolution, ray_heights, max_nodes=5000000):
    """Fixed +1um grid-origin shift; explicit directional retries, never zero fill."""
    import numpy as np
    epsilon = 1e-6
    origin = np.floor(np.asarray(bounds[:2]) / resolution) * resolution + epsilon
    count = np.ceil((np.asarray(bounds[2:]) - origin) / resolution).astype(int) + 1
    if (count < 2).any() or int(count.prod()) > max_nodes:
        raise ValueError(f'Invalid/oversized fine grid: {tuple(count)}; max_nodes={max_nodes}')
    heights = np.empty((count[1], count[0]), dtype=np.float32)
    retries = 0
    xs = origin[0] + np.arange(count[0]) * resolution
    for row in range(count[1]):
        xy = np.column_stack((xs, np.full(len(xs), origin[1] + row * resolution)))
        z = ray_heights(xy)
        bad = ~np.isfinite(z)
        retries += int(bad.sum())
        for offset in ((epsilon, 0), (0, epsilon), (-epsilon, 0), (0, -epsilon)):
            if not bad.any():
                break
            z[bad] = ray_heights(xy[bad] + np.asarray(offset))
            bad = ~np.isfinite(z)
        if bad.any():
            raise ValueError(f'Unresolved ray misses at row {row}: {int(bad.sum())}; refusing zero fill')
        heights[row] = z
        if row % 100 == 0 or row == count[1] - 1:
            print(f'fine grid row {row+1}/{count[1]} ({int(count.prod())} nodes total)', flush=True)
    return tr.HeightField(torch.from_numpy(heights), tuple(origin.tolist()), resolution), dict(
        origin_shift_m=epsilon, no_hit_nodes_retried=retries, unresolved_no_hits=0,
        nodes=int(count.prod()), shape=list(heights.shape),
        boundary_rule='origin shifted +1um in x/y; missing rays retry +x,+y,-x,-y by 1um; never fill zero')
