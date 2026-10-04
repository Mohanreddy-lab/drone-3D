import numpy as np
import pytest

from drone3d import telemetry

MODERN = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2023-05-12 10:21:33.123
[iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 17.385044] [longitude: 78.486671] [rel_alt: 40.000 abs_alt: 560.123] </font>

2
00:00:01,000 --> 00:00:01,033
<font size="28">FrameCnt: 31, DiffTime: 33ms
[focal_len : 24.00] [latitude: 17.385089] [longitude: 78.486671] [rel_alt: 41.000 abs_alt: 561.123] [gb_yaw: 12.5 gb_pitch: -80.0 gb_roll: 0.0] </font>
"""

OLD = """1
00:00:00,000 --> 00:00:01,000
HOME(78.4866,17.3850) 2018.01.01 12:00:00
GPS(78.486671,17.385044,19) BAROMETER:40.1m

2
00:00:01,000 --> 00:00:02,000
HOME(78.4866,17.3850) 2018.01.01 12:00:01
GPS(78.486771,17.385044,19) BAROMETER:40.3m
"""


def test_modern_dji_srt(tmp_path):
    p = tmp_path / "a.SRT"
    p.write_text(MODERN)
    tel = telemetry.load(p)
    assert len(tel) == 2
    assert tel.lat[1] == pytest.approx(17.385089)
    assert tel.alt.tolist() == [40.0, 41.0]
    assert tel.focal_35mm == pytest.approx(24.0)
    assert tel.gimbal_pitch is not None and tel.gimbal_pitch[1] == pytest.approx(-80.0)
    enu = tel.enu()
    assert enu[1, 1] == pytest.approx(5.0, abs=0.1)  # 0.000045 deg latitude ~ 5 m north


def test_old_dji_srt(tmp_path):
    p = tmp_path / "b.srt"
    p.write_text(OLD)
    tel = telemetry.load(p)
    assert len(tel) == 2
    assert tel.alt.tolist() == [40.1, 40.3]
    assert tel.enu()[1, 0] == pytest.approx(10.6, abs=0.2)  # 0.0001 deg longitude east


def test_csv_and_interpolation(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("time,latitude,longitude,altitude,yaw\n0,17.0,78.0,30,0\n2,17.0,78.0,40,90\n")
    tel = telemetry.load(p)
    pos = tel.positions_at(np.array([1.0]))
    assert pos[0, 2] == pytest.approx(35.0, abs=1e-3)
    pos_late = tel.positions_at(np.array([1.0]), offset=0.5)
    assert pos_late[0, 2] == pytest.approx(37.5, abs=1e-3)


def test_gps_spike_is_dropped(tmp_path):
    rows = ["time,lat,lon,alt"]
    for i in range(10):
        lat = 17.0 + i * 1e-5 + (0.01 if i == 5 else 0)  # 1 km jump at t=5
        rows.append(f"{i},{lat},78.0,40")
    p = tmp_path / "s.csv"
    p.write_text("\n".join(rows))
    tel = telemetry.load(p)
    assert len(tel) == 9 and tel.notes


def test_no_gps_lock_is_an_error(tmp_path):
    p = tmp_path / "c.srt"
    p.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[latitude: 0.0] [longitude: 0.0] [rel_alt: 0.0 abs_alt: 0]\n"
    )
    with pytest.raises(ValueError):
        telemetry.load(p)
