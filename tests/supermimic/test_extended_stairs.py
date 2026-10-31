"""Harder stairs extend the locomotion curriculum without changing old tasks."""
import pytest

from lite3_supermimic.locomotion.env import make_env_cfg
from lite3_supermimic.locomotion.train import TrainCfg


def test_extended_stairs_keep_old_rows_and_add_harder_rows():
    original = make_env_cfg("terrain")
    extended = make_env_cfg("terrain", harder_stairs=True)
    old = original.scene.terrain.terrain_generator
    new = extended.scene.terrain.terrain_generator
    assert old.num_rows == 5
    assert new.num_rows == 9
    assert original.scene.terrain.max_init_terrain_level == 0
    assert extended.scene.terrain.max_init_terrain_level == 4
    for name in ("stairs", "stairs_inv"):
        assert old.sub_terrains[name].step_height_range == (0.02, 0.10)
        assert new.sub_terrains[name].step_height_range == (0.02, 0.18)
        for row in range(5):
            before = 0.02 + row / 4 * (0.10 - 0.02)
            after = 0.02 + row / 8 * (0.18 - 0.02)
            assert after == pytest.approx(before)
    assert make_env_cfg("terrain").scene.terrain.terrain_generator.num_rows == 5


def test_extended_stairs_only_allowed_for_terrain():
    with pytest.raises(ValueError, match="harder-stairs"):
        TrainCfg(stage="flat", harder_stairs=True)
    assert TrainCfg(stage="terrain", harder_stairs=True).harder_stairs
