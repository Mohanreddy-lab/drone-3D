import numpy as np

from drone3d.frames import choose, footprint


def test_spacing_follows_progress():
    progress = np.linspace(0, 100, 1001)  # 100 m at 10 cm per frame
    sharp = np.ones(1001)
    idx = choose(progress, 10.0, sharp)
    gaps = np.diff(progress[idx])
    assert 9 <= len(idx) <= 12
    assert gaps.min() >= 9.99 and gaps.max() <= 13.0


def test_blurred_frames_are_skipped():
    rng = np.random.default_rng(0)
    progress = np.linspace(0, 100, 1001)
    sharp = 100 + rng.normal(0, 2, 1001)
    blurred = np.arange(0, 1001, 3)
    sharp[blurred] = 20.0  # every third frame is motion blurred
    idx = choose(progress, 5.0, sharp)
    assert not set(idx) & set(blurred.tolist())


def test_hovering_adds_no_frames():
    progress = np.concatenate(
        [np.linspace(0, 20, 200), np.full(500, 20.0), np.linspace(20, 40, 200)]
    )
    idx = choose(progress, 5.0, np.ones(len(progress)))
    assert len(idx) <= 10


def test_turn_triggers_keyframe():
    progress = np.linspace(0, 10, 101)
    yaw = np.where(np.arange(101) > 50, 90.0, 0.0)
    idx = choose(progress, 100.0, np.ones(101), yaw=yaw, yaw_step=20)
    assert len(idx) >= 2


def test_footprint_nadir():
    # 40 m altitude, 60 deg FOV both ways: 2 * 40 * tan(30) = 46.2 m
    fp = footprint(40.0, np.radians(60), np.radians(60), -90.0)
    assert abs(fp - 46.19) < 0.1
