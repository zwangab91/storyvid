"""Keyed gaits: walk, run, bound and flap cycles for cut-paper figures.

A sine swing makes every limb pendulum the same way, so a puppet's run reads as a jumping jack. A gait here is
one cycle of keyed angles per joint, taken from how people and animals actually move (a leg plants and pushes
back slowly, then swings forward fast; the knee folds during the swing; the body is lowest when both feet are
down), with Catmull-Rom curves between the keys.

Angles are degrees for a limb hanging straight down from a body that faces right: positive turns the limb
backward (clockwise on screen), negative swings it forward. Knees and elbows are relative to the limb above.
"y" keys are a vertical bob in units of the figure's leg length (negative is up); "rot" keys on the body lean it.

The cycle is paced by travel, not a clock: at its natural speed (`stride` leg lengths per `period`) a figure's
feet keep pace with the ground; slower or faster, the cadence changes only a little (as in life, speed comes from
stride length) so a storybook's unhurried moves never play in slow motion; as the figure stops the gait settles. A figure that runs on the
spot (the scenery scrolls instead) keeps the fixed `period` with "in_place": true on its rig window.
"""
from __future__ import annotations

import bisect

# ------------------------------------------------------------------ joint curves (phase 0 = this foot lands)
HUMAN_WALK = {
    "hip": [(0, -22), (0.12, -17), (0.32, -3), (0.5, 11), (0.6, 7), (0.72, -10), (0.86, -25)],
    "knee": [(0, 3), (0.13, 14), (0.38, 4), (0.55, 20), (0.68, 56), (0.8, 38), (0.93, 6)],
    "shoulder": [(0, 15), (0.5, -15)],
    "elbow": [(0, -8), (0.5, -26)],
    "bob": [(0, 0.012), (0.25, -0.012), (0.5, 0.012), (0.75, -0.012)],
    "head": [(0, 1.5), (0.25, -1), (0.5, 1.5), (0.75, -1)],
}
HUMAN_RUN = {
    "hip": [(0, -30), (0.14, -10), (0.36, 24), (0.48, 18), (0.66, -26), (0.82, -44)],
    "knee": [(0, 16), (0.14, 36), (0.36, 16), (0.52, 92), (0.68, 108), (0.84, 52), (0.95, 20)],
    "shoulder": [(0, 38), (0.5, -38)],
    "elbow": [(0, -70), (0.25, -84), (0.5, -96), (0.75, -84)],
    "bob": [(0, 0.01), (0.14, 0.035), (0.4, -0.04), (0.5, 0.01), (0.64, 0.035), (0.9, -0.04)],
    "lean": [(0, -9), (0.25, -7), (0.5, -9), (0.75, -7)],
    "head": [(0, 4), (0.25, 2), (0.5, 4), (0.75, 2)],
}
# four legs, lateral-sequence walk: hind-near, fore-near, hind-far, fore-far a quarter cycle apart
QUAD_WALK = {
    "hind": [(0, -16), (0.62, 14), (0.78, 4), (0.9, -14)],
    "fore": [(0, -14), (0.62, 16), (0.76, 2), (0.9, -16)],
    "bob": [(0, 0.015), (0.25, -0.015), (0.5, 0.015), (0.75, -0.015)],
    "head": [(0, 2), (0.25, -2), (0.5, 2), (0.75, -2)],
    "tail": [(0, 6), (0.5, -6)],
}
# gallop: hind legs drive together, fore legs reach and land together; the body stretches in flight
QUAD_RUN = {
    "hind": [(0, -34), (0.12, -14), (0.3, 42), (0.42, 48), (0.66, 4), (0.84, -38)],
    "fore": [(0, 30), (0.16, 44), (0.4, -10), (0.56, -44), (0.7, -36), (0.84, 8)],
    "bob": [(0, 0.06), (0.18, 0.08), (0.4, -0.16), (0.6, 0.02), (0.78, 0.05), (0.92, -0.04)],
    "pitch": [(0, 5), (0.2, 7), (0.42, -6), (0.62, -3), (0.8, 3)],
    "head": [(0, -3), (0.3, 4), (0.6, -2)],
    "tail": [(0, 10), (0.4, -14), (0.7, 6)],
    "ears": [(0, -6), (0.35, 12), (0.7, -2)],
}
# a hare's bound: one hind leg piece and one fore leg piece in the art
HARE_BOUND = {
    "hind": [(0, -30), (0.1, -20), (0.32, 46), (0.44, 56), (0.7, 10), (0.88, -34)],
    "fore": [(0, 34), (0.22, 50), (0.46, -24), (0.6, -48), (0.74, -30), (0.88, 12)],
    "bob": [(0, 0.1), (0.16, 0.12), (0.42, -0.32), (0.62, 0.02), (0.8, 0.08)],
    "pitch": [(0, 8), (0.18, 10), (0.44, -8), (0.66, -4), (0.84, 5)],
    "ears": [(0, -8), (0.4, 16), (0.7, 0)],
    "head": [(0, -3), (0.4, 4), (0.7, -1)],
}
# a slow tortoise plod: lateral sequence, long stance, the head nods with each step
PLOD = {
    "hind": [(0, -12), (0.7, 10), (0.84, 2), (0.94, -10)],
    "fore": [(0, -14), (0.7, 12), (0.84, 0), (0.94, -12)],
    "bob": [(0, 0.03), (0.25, -0.02), (0.5, 0.03), (0.75, -0.02)],
    "head": [(0, -4), (0.25, 3), (0.5, -4), (0.75, 3)],
}
# a wing beat (for wings painted raised): a fast, strong downstroke and a slower recovery; the body rises as
# the wings come down
FLAP = {
    "wing_near": [(0, 32), (0.36, -34), (0.52, -22), (0.8, 24)],
    "wing_far": [(0, 26), (0.38, -26), (0.54, -16), (0.82, 20)],
    "bob": [(0, 0.0), (0.36, -0.18), (0.7, 0.06)],
    "tail": [(0, 4), (0.36, -6), (0.7, 2)],
    "pitch": [(0, 0), (0.36, -3), (0.7, 2)],
}

# ------------------------------------------------------------------ templates: role -> (curve, channel, phase)
# stride: leg lengths travelled per cycle; period: seconds per cycle when running on the spot
TEMPLATES = {
    "biped_walk": {"curves": HUMAN_WALK, "stride": 1.7, "period": 1.05, "entries": [
        ("thigh_n", "hip", "rot", 0.0), ("thigh_f", "hip", "rot", 0.5),
        ("shin_n", "knee", "rot", 0.0), ("shin_f", "knee", "rot", 0.5),
        ("uarm_n", "shoulder", "rot", 0.0), ("uarm_f", "shoulder", "rot", 0.5),
        ("farm_n", "elbow", "rot", 0.0), ("farm_f", "elbow", "rot", 0.5),
        ("body", "bob", "y", 0.0), ("head", "head", "rot", 0.0)]},
    "biped_run": {"curves": HUMAN_RUN, "stride": 3.0, "period": 0.62, "entries": [
        ("thigh_n", "hip", "rot", 0.0), ("thigh_f", "hip", "rot", 0.5),
        ("shin_n", "knee", "rot", 0.0), ("shin_f", "knee", "rot", 0.5),
        ("uarm_n", "shoulder", "rot", 0.0), ("uarm_f", "shoulder", "rot", 0.5),
        ("farm_n", "elbow", "rot", 0.0), ("farm_f", "elbow", "rot", 0.5),
        ("body", "bob", "y", 0.0), ("body", "lean", "rot", 0.0), ("head", "head", "rot", 0.0)]},
    "quad_walk": {"curves": QUAD_WALK, "stride": 2.2, "period": 0.9, "entries": [
        ("hind_n", "hind", "rot", 0.0), ("fore_n", "fore", "rot", 0.25),
        ("hind_f", "hind", "rot", 0.5), ("fore_f", "fore", "rot", 0.75),
        ("body", "bob", "y", 0.0), ("head", "head", "rot", 0.0), ("tail", "tail", "rot", 0.0)]},
    "quad_run": {"curves": QUAD_RUN, "stride": 4.2, "period": 0.5, "entries": [
        ("hind_n", "hind", "rot", 0.0), ("hind_f", "hind", "rot", 0.06),
        ("fore_n", "fore", "rot", 0.0), ("fore_f", "fore", "rot", 0.07),
        ("body", "bob", "y", 0.0), ("body", "pitch", "rot", 0.0), ("head", "head", "rot", 0.0),
        ("tail", "tail", "rot", 0.0), ("ears", "ears", "rot", 0.0)]},
    "hare_bound": {"curves": HARE_BOUND, "stride": 5.0, "period": 0.42, "entries": [
        ("hind_n", "hind", "rot", 0.0), ("fore_n", "fore", "rot", 0.0),
        ("body", "bob", "y", 0.0), ("body", "pitch", "rot", 0.0),
        ("ears", "ears", "rot", 0.0), ("head", "head", "rot", 0.0)]},
    "plod": {"curves": PLOD, "stride": 1.6, "period": 1.5, "entries": [
        ("hind_n", "hind", "rot", 0.0), ("fore_n", "fore", "rot", 0.25),
        ("hind_f", "hind", "rot", 0.5), ("fore_f", "fore", "rot", 0.75),
        ("body", "bob", "y", 0.0), ("head", "head", "rot", 0.0)]},
    "flap": {"curves": FLAP, "stride": 0.0, "period": 0.46, "entries": [
        ("wing_near", "wing_near", "rot", 0.0), ("wing_far", "wing_far", "rot", 0.04),
        ("body", "bob", "y", 0.0), ("body", "pitch", "rot", 0.0), ("tail", "tail", "rot", 0.0)]},
}


def curve_value(keys, ph: float) -> float:
    """Periodic Catmull-Rom through [(phase, value), ...] keys in [0, 1); ph wraps."""
    n = len(keys)
    if n == 1:
        return float(keys[0][1])
    ph %= 1.0
    xs = [k[0] for k in keys]
    i = bisect.bisect_right(xs, ph) - 1  # segment keys[i] -> keys[i+1] (wrapping)

    def key(j):
        x, v = keys[j % n]
        return x + (j // n), v  # unwrap phase across the cycle boundary
    x0, v0 = key(i - 1)
    x1, v1 = key(i)
    x2, v2 = key(i + 1)
    x3, v3 = key(i + 2)
    seg = x2 - x1
    f = (ph - x1) / seg if seg > 1e-9 else 0.0
    m1 = (v2 - v0) / max(x2 - x0, 1e-9) * seg
    m2 = (v3 - v1) / max(x3 - x1, 1e-9) * seg
    f2, f3 = f * f, f * f * f
    return (2 * f3 - 3 * f2 + 1) * v1 + (f3 - 2 * f2 + f) * m1 + (-2 * f3 + 3 * f2) * v2 + (f3 - f2) * m2


def build(template: str, parts: dict, leg: float, scale: dict | None = None, extra: list | None = None) -> list:
    """-> part anim entries for a gait. `parts` maps roles (thigh_n, hind_f, ears, ...) to part ids (a role may
    map to a list); roles the figure lacks are skipped. `leg` is the leg length in actor units. `scale` multiplies
    a role's curve (by role, e.g. "farm_n", or by curve, e.g. "elbow"); [curve, factor] swaps the curve too."""
    T = TEMPLATES[template]
    scale = scale or {}
    out = []
    entries = [tuple(e) for e in T["entries"]]
    for role, curve, channel, phase, factor in extra or []:  # e.g. a basket that hangs straight: ["basket",
        entries.append((role, curve, channel, phase))         # "shoulder", "rot", 0, -1]
        scale = {**scale, role: [curve, factor]}
    for role, curve, channel, phase in entries:
        ids = parts.get(role)
        if not ids:
            continue
        sk = scale.get(role, scale.get(curve, 1.0))
        if isinstance(sk, list):  # [curve, factor]: drive this role with another joint's curve (a forearm holding
            curve, sk = sk        # a staff turns against the shoulder, so the staff stays upright)
        k = float(sk) * (leg if channel == "y" else 1.0)
        out.append({"part": ids, "type": "gait", "channel": channel, "phase": phase,
                    "keys": [[round(x, 4), round(v * k, 4)] for x, v in T["curves"][curve]],
                    "stride": round(T["stride"] * leg, 2), "period": T["period"]})
    return out
