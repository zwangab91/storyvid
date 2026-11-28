"""Keyed gaits: curves pass through their keys, wrap smoothly, and are paced by distance travelled."""
import numpy as np

from storyvid import anim, gait
from storyvid.render import travel_table, pace_gaits
from storyvid.rig import prepare_piece


def test_curves_hit_keys_and_wrap():
    for T in gait.TEMPLATES.values():
        for keys in T["curves"].values():
            for x, v in keys:
                assert abs(gait.curve_value(keys, x) - v) < 1e-6
            assert abs(gait.curve_value(keys, 0.99999) - gait.curve_value(keys, 0.0)) < 0.05 * (1 + abs(keys[0][1]))


def test_travel_paces_the_cycle():
    item = {"at": [100, 500], "anims": [{"type": "move", "to": [1000, 500], "t": [0, 1], "ease": "linear"}]}
    us, ds, vs = travel_table(item, 6.0)
    assert abs(ds[-1] - 900) < 1 and abs(vs[len(vs) // 2] - 150) < 2
    e = {"type": "gait", "keys": [[0, 0], [0.25, 10], [0.5, 0], [0.75, -10]], "stride": 300, "period": 2.0}
    paced = pace_gaits({0: [e]}, item, 6.0)[0][0]  # 150 units/s is exactly the natural pace: 3 cycles
    assert abs(paced["_cyc"][1][-1] - 3.0) < 0.05
    slow = pace_gaits({0: [dict(e, period=0.5)]}, item, 6.0)[0][0]  # a quarter of the natural pace
    assert 12 * 0.6 < slow["_cyc"][1][-1] < 12 * 0.95  # cadence slows a little, not to a quarter


def test_gait_settles_when_the_figure_stops_and_runs_on_the_spot_when_asked():
    item = {"at": [100, 500], "anims": [{"type": "move", "to": [1000, 500], "t": [0, 0.5]}]}
    e = {"type": "gait", "keys": [[0, 5], [0.5, -5]], "stride": 300, "period": 1.0}
    paced = pace_gaits({0: [e]}, item, 6.0)[0][0]
    assert anim.part_anim_state([paced], 0.8, 6.0)["rot"] == 0.0
    spot = pace_gaits({0: [dict(e, in_place=True)]}, {"at": [0, 0]}, 6.0)[0][0]
    assert "_dist" not in spot and anim.part_anim_state([spot], 0.0, 6.0)["rot"] == 5.0


def test_build_maps_roles_and_scales():
    out = gait.build("biped_walk", {"thigh_n": "a", "body": "b"}, 200.0, {"thigh_n": 0.5})
    assert {e["part"] for e in out} == {"a", "b"}
    hip = next(e for e in out if e["part"] == "a")
    assert hip["keys"][0][1] == round(gait.HUMAN_WALK["hip"][0][1] * 0.5, 4) and hip["stride"] == 340.0
    bob = next(e for e in out if e["part"] == "b")
    assert bob["channel"] == "y" and bob["keys"][0][1] == round(gait.HUMAN_WALK["bob"][0][1] * 200, 4)


def test_caps_round_a_limb_end_about_its_joint():
    img = np.zeros((100, 140, 4), np.uint8)
    img[30:70, 20:120] = 255
    for side, corner in (("left", (30, 20)), ("right", (30, 119))):
        out, pv = prepare_piece(img, {"cap": side})
        a = out[..., 3] > 127
        assert a[int(pv[1]), int(pv[0])] and not a[corner]
