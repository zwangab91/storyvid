"""Animation: easing curves and evaluation of item / part animation lists.

Times in animation specs are FRACTIONS of the scene duration ("t": [0.2, 0.7]),
so storyboards stay valid whatever the narration length. Periods are in seconds.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import spline
from .gait import curve_value


def _back(p, k=1.70158):
    p -= 1
    return p * p * ((k + 1) * p + k) + 1


EASE = {
    "linear": lambda p: p,
    "in": lambda p: p * p * p,
    "out": lambda p: 1 - (1 - p) ** 3,
    "inout": lambda p: 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2,
    "sine": lambda p: 0.5 - 0.5 * math.cos(math.pi * p),
    "back": _back,
    "bounce": lambda p: 1 - abs(math.cos(p * math.pi * 2.5)) * (1 - p) ** 2,
}


def _prog(a, u):
    t0, t1 = a.get("t", (0.0, 1.0))
    if t1 <= t0:
        return 1.0 if u >= t0 else 0.0
    return min(1.0, max(0.0, (u - t0) / (t1 - t0)))


def _ease(a, p, default="inout"):
    return EASE.get(a.get("ease", default), EASE["inout"])(p)


def _env(a, u, dur):
    """Amplitude envelope for continuous anims with an optional time window."""
    if "t" not in a:
        return 1.0, 0.0
    t0, t1 = a["t"]
    if u < t0 or u > t1:
        return 0.0, t0 * dur
    ramp = max(1e-3, min(0.25, (t1 - t0) * dur / 4)) / max(dur, 1e-3)
    e = min(1.0, (u - t0) / ramp, (t1 - u) / ramp)
    e = e * e * (3 - 2 * e)
    return e, t0 * dur


SIDES = {"left": (-350, None), "right": (2270, None), "top": (None, -350), "bottom": (None, 1430)}


def _side_point(spec, cur):
    if isinstance(spec, str):
        sx, sy = SIDES[spec]
        return [cur[0] if sx is None else sx, cur[1] if sy is None else sy]
    return list(spec)


def _det_noise(s, seed):
    k = int(s * 12)
    r = np.random.default_rng((seed * 7919 + k) & 0xFFFFFFFF)
    return r.uniform(-1, 1, 2)


def new_state(x=0.0, y=0.0, scale=1.0, rot=0.0, flip=False, opacity=1.0):
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    return {"x": float(x), "y": float(y), "sx": float(sx), "sy": float(sy), "rot": float(rot),
            "op": float(opacity), "flip": bool(flip), "vis": True}


def apply_anims(state: dict, anims: list, u: float, dur: float, seed: int = 0) -> dict:
    """Apply a list of animation specs to a transform state at scene fraction u."""
    s = u * dur
    st = dict(state)
    for a in anims or []:
        typ = a.get("type")
        if typ == "move":
            p = _ease(a, _prog(a, u))
            tx, ty = a["to"]
            st["x"] += (tx - st["x"]) * p
            st["y"] += (ty - st["y"]) * p
        elif typ == "by":  # relative move
            p = _ease(a, _prog(a, u))
            st["x"] += a["d"][0] * p
            st["y"] += a["d"][1] * p
        elif typ == "path":
            p = _ease(a, _prog(a, u), "linear")
            pts = [[st["x"], st["y"]]] + [list(q) for q in a["points"]]
            P = spline(pts, closed=False) if a.get("smooth", True) and len(pts) > 2 else np.array(pts, float)
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            L = p * cum[-1]
            i = int(np.clip(np.searchsorted(cum, L) - 1, 0, len(seg) - 1))
            f = (L - cum[i]) / max(seg[i], 1e-9)
            q = P[i] + (P[i + 1] - P[i]) * f
            st["x"], st["y"] = float(q[0]), float(q[1])
            if a.get("orient"):
                d = P[i + 1] - P[i]
                st["rot"] += math.degrees(math.atan2(d[1], d[0])) * (-1 if st["flip"] else 1)
        elif typ == "enter":
            p = _ease(a, _prog(a, u), "out")
            fx, fy = _side_point(a.get("from", "left"), (st["x"], st["y"]))
            st["x"] = fx + (st["x"] - fx) * p
            st["y"] = fy + (st["y"] - fy) * p
        elif typ == "exit":
            p = _ease(a, _prog(a, u), "in")
            tx, ty = _side_point(a.get("to", "right"), (st["x"], st["y"]))
            st["x"] += (tx - st["x"]) * p
            st["y"] += (ty - st["y"]) * p
        elif typ == "scale":
            p = _ease(a, _prog(a, u))
            to = a["to"]
            tx, ty = (to, to) if np.isscalar(to) else to
            st["sx"] += (tx - st["sx"]) * p
            st["sy"] += (ty - st["sy"]) * p
        elif typ == "rotate":
            p = _ease(a, _prog(a, u))
            st["rot"] += (a["to"] - st["rot"]) * p
        elif typ == "fade":
            p = _ease(a, _prog(a, u), "sine")
            st["op"] += (a.get("to", 0.0) - st["op"]) * p
        elif typ in ("pop", "grow"):
            t0 = a.get("t", (0, 0.1))[0]
            if u < t0:
                st["vis"] = False
            p = _prog(a, u)
            k = _back(p) if typ == "pop" else EASE["out"](p)
            st["sx"] *= k
            st["sy"] *= k
            st["op"] *= min(1.0, p * 4)
        elif typ == "vanish":
            p = _ease(a, _prog(a, u), "in")
            st["sx"] *= 1 - p
            st["sy"] *= 1 - p
            st["op"] *= 1 - p
        elif typ == "show":
            t0, t1 = a["t"]
            if u < t0 or u > t1:
                st["vis"] = False
            else:
                fd = a.get("fade", 0.25) / max(dur, 1e-3)
                st["op"] *= min(1.0, (u - t0) / fd if t0 > 0 else 1.0, (t1 - u) / fd if t1 < 1 else 1.0)
        elif typ == "nudge":  # a gesture that goes and comes back, added on top of whatever else moves the part
            p = _prog(a, u)    # (a staff lifted while walking): "deg" and/or "d" at its peak, mid-window
            if 0.0 < p < 1.0:
                k = math.sin(math.pi * p) ** float(a.get("sharp", 1.0))
                st["rot"] += float(a.get("deg", 0.0)) * k
                d = a.get("d", (0, 0))
                st["x"] += d[0] * k
                st["y"] += d[1] * k
        elif typ == "flip":  # turn round at "at" (a scene fraction), or on the words (its "t" from a beat)
            if u >= a.get("at", a["t"][0] if "t" in a else 0.5):
                st["flip"] = not st["flip"]
        elif typ == "float_up":  # looping rise-and-fade (Zzz, bubbles, sparks)
            e, s0 = _env(a, u, dur)
            if e <= 0:
                st["vis"] = False
                continue
            per = float(a.get("period", 2.0))
            fr = ((s - s0) / per + a.get("phase", 0.0)) % 1.0
            d = a.get("dir", (0.25, -1))
            st["x"] += d[0] * a.get("dist", 150) * fr
            st["y"] += d[1] * a.get("dist", 150) * fr
            st["op"] *= math.sin(math.pi * fr) * e
            k = 0.6 + 0.6 * fr
            st["sx"] *= k
            st["sy"] *= k
        elif typ in ("hop", "bob", "sway", "spin", "pulse", "shake", "squash", "swing", "drift"):
            e, s0 = _env(a, u, dur)
            if e <= 0:
                continue
            ss = s - s0
            per = float(a.get("period", 1.0))
            ph = float(a.get("phase", 0.0))
            w = 2 * math.pi * (ss / per + ph)
            if typ == "hop":
                st["y"] -= a.get("height", 40) * abs(math.sin(w / 2)) * e
            elif typ == "bob":
                d = a.get("dir", (0, 1))
                v = a.get("amp", 10) * math.sin(w) * e
                st["x"] += d[0] * v
                st["y"] += d[1] * v
            elif typ == "drift":  # slow random wander, good for clouds
                st["x"] += a.get("amp", 20) * math.sin(w * 0.37 + 1.3) * e
                st["y"] += a.get("amp", 20) * 0.4 * math.sin(w * 0.61) * e
            elif typ in ("sway", "swing"):
                st["rot"] += a.get("deg", 8) * math.sin(w) * e
            elif typ == "spin":
                st["rot"] += a.get("speed", 30) * ss * e
            elif typ == "pulse":
                k = 1 + a.get("amp", 0.05) * math.sin(w) * e
                st["sx"] *= k
                st["sy"] *= k
            elif typ == "squash":
                k = a.get("amp", 0.06) * math.sin(w) * e
                st["sx"] *= 1 + k
                st["sy"] *= 1 - k
            elif typ == "shake":
                n = _det_noise(s, seed + 17)
                st["x"] += a.get("amp", 8) * n[0] * e
                st["y"] += a.get("amp", 8) * n[1] * e
        elif typ == "gait":  # keyed cycle (see gait.py), paced by how fast the item travels
            e, s0 = _env(a, u, dur)
            if e <= 0:
                continue
            ph0 = float(a.get("phase", 0.0))
            cyc = a.get("_cyc")
            if cyc is not None and not a.get("in_place"):
                us, cs, ks = cyc
                ph = float(np.interp(u, us, cs)) + ph0
                e *= float(np.interp(u, us, ks))
            else:
                ph = (s - s0) / float(a.get("period", 1.0)) + ph0
            val = curve_value(a["keys"], ph) * e * float(a.get("amp", 1.0))
            ch = a.get("channel", "rot")
            if ch == "rot":
                st["rot"] += val
            elif ch == "y":
                st["y"] += val
            elif ch == "x":
                st["x"] += val
        elif typ in (None, "none"):
            pass
        else:
            raise ValueError(f"unknown animation type {typ!r}")
    return st


def part_anim_state(anims: list, u: float, dur: float, seed: int = 0):
    """Part-level rig anims -> (dx, dy, rot, sx, sy, opacity, visible)."""
    st = new_state()
    s = u * dur
    for a in anims:
        typ = a.get("type")
        if typ == "blink":
            per = a.get("period", 3.2)
            ph = a.get("phase", 0.37) * per
            k = (s + ph) % per
            if k < a.get("dur", 0.14):
                st["sy"] *= 0.12
            continue
        st = apply_anims(st, [a], u, dur, seed)
    return st


def state_matrix(st: dict) -> np.ndarray:
    """Translate * Rotate * Scale (* mirror) as a 3x3 matrix."""
    a = math.radians(st["rot"])
    c, s = math.cos(a), math.sin(a)
    sx = -st["sx"] if st["flip"] else st["sx"]
    sy = st["sy"]
    return np.array([[c * sx, -s * sy, st["x"]], [s * sx, c * sy, st["y"]], [0, 0, 1]], np.float64)


def T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
