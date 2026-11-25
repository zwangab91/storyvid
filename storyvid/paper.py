"""Procedural painted tissue-paper textures.

Each "paint" is a recipe of a few colours plus a brush technique
(streaks, swirl, dabs, cells, speckle, stripes, dots, sponge, flat).
Textures are float32 RGB in [0, 1], generated deterministically from the
recipe + seed and cached in memory and on disk.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os

import cv2
import numpy as np

# --------------------------------------------------------------------------- colours

def hex2rgb(c) -> np.ndarray:
    if isinstance(c, (list, tuple, np.ndarray)):
        a = np.asarray(c, np.float32)
        return a / 255.0 if a.max() > 1.0 else a
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def rgb2hex(rgb) -> str:
    r, g, b = [int(round(max(0, min(1, x)) * 255)) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def shade(c, dl=0.0, ds=0.0, dh=0.0):
    """Shift a colour in HLS space (dl, ds in [-1, 1], dh in degrees)."""
    r, g, b = hex2rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + dh / 360.0) % 1.0
    l = min(1, max(0, l + dl))
    s = min(1, max(0, s + ds))
    return rgb2hex(colorsys.hls_to_rgb(h, l, s))


def auto_palette(base: str) -> list[str]:
    """Derive streak colours from a single base colour, like a painter would."""
    return [base, shade(base, 0.14, -0.05, -8), shade(base, -0.16, 0.05, 10),
            shade(base, 0.26, -0.1, -18), shade(base, -0.06, 0.1, 22)]


# --------------------------------------------------------------------------- noise helpers

def _norm01(a):
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def smooth_noise(h, w, sigma, rng):
    """Isotropic gaussian-filtered noise, zero mean, unit std. Cheap for big sigma."""
    f = max(1, int(sigma / 3))
    hh, ww = max(4, h // f + 4), max(4, w // f + 4)
    n = rng.standard_normal((hh, ww)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(0.5, sigma / f))
    n = cv2.resize(n, (ww * f, hh * f), interpolation=cv2.INTER_CUBIC)[:h, :w]
    return (n - n.mean()) / (n.std() + 1e-6)


def fbm(h, w, sigma, rng, octaves=4, gain=0.5):
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        acc += amp * smooth_noise(h, w, max(0.6, sigma / (2 ** o)), rng)
        tot += amp
        amp *= gain
    return acc / tot


def aniso_noise(S, angle_deg, sx, sy, rng):
    """Noise stretched along `angle` — looks like brush/bristle marks."""
    P = int(S * 1.45) + 8
    f = max(1, int(sy / 2))
    n = rng.standard_normal((P // f + 2, P // f + 2)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=max(0.4, sx / f), sigmaY=max(0.4, sy / f))
    if f > 1:
        n = cv2.resize(n, (P, P), interpolation=cv2.INTER_CUBIC)
    n = n[:P, :P]
    M = cv2.getRotationMatrix2D((P / 2, P / 2), angle_deg, 1.0)
    n = cv2.warpAffine(n, M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    o = (P - S) // 2
    n = n[o:o + S, o:o + S]
    return (n - n.mean()) / (n.std() + 1e-6)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def warp(img, ax, ay):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + ax, gy + ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- techniques

def _layer(base, col, mask):
    return base * (1 - mask[..., None]) + col[None, None, :] * mask[..., None]


def _streak_field(S, angle, scale, rng, jitter=14):
    """A field in ~[-2, 2] made of long overlapping brush strokes."""
    a = aniso_noise(S, angle + rng.uniform(-jitter, jitter), 55 * scale, 4.0 * scale, rng)
    b = aniso_noise(S, angle + rng.uniform(-jitter, jitter), 110 * scale, 9.0 * scale, rng)
    return 0.6 * a + 0.4 * b


def paint_texture(spec: dict, S: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = spec.get("colors") or [spec.get("color", "#888888")]
    if len(cols) == 1:
        cols = auto_palette(cols[0])
    cols = [hex2rgb(c) for c in cols]
    style = spec.get("style", "streaks")
    angle = float(spec.get("angle", rng.uniform(-35, 35)))
    scale = float(spec.get("scale", 1.0))
    contrast = float(spec.get("contrast", 1.0))
    base = np.ones((S, S, 3), np.float32) * cols[0]

    if style == "flat":
        img = base.copy()
        v = fbm(S, S, 90 * scale, rng, 3)
        img *= (1 + 0.035 * contrast * v)[..., None]

    elif style in ("streaks", "swirl"):
        img = base.copy()
        if style == "swirl":
            amp = 70 * scale
            wx = smooth_noise(S, S, 110 * scale, rng) * amp
            wy = smooth_noise(S, S, 110 * scale, rng) * amp
        for i, c in enumerate(cols[1:]):
            f = _streak_field(S, angle, scale, rng)
            low = smooth_noise(S, S, 160 * scale, rng)
            fld = f + 0.9 * low
            if style == "swirl":
                fld = warp(fld, wx, wy)
            th = 0.55 + 0.25 * i  # later colours are rarer accents
            m = smoothstep(th - 0.35, th + 0.25, fld) * min(1.0, 0.92 * contrast)
            img = _layer(img, c, m)

    elif style == "dabs":
        img = base.copy()
        for i, c in enumerate(cols[1:]):
            fld = fbm(S, S, 38 * scale, rng, 3) + 0.5 * _streak_field(S, angle, 0.5 * scale, rng)
            th = 0.65 + 0.2 * i
            m = smoothstep(th - 0.12, th + 0.12, fld) * min(1.0, 0.95 * contrast)
            img = _layer(img, c, m)

    elif style == "sponge":
        img = base.copy()
        for i, c in enumerate(cols[1:]):
            fld = fbm(S, S, 6 * scale, rng, 3) + 0.8 * smooth_noise(S, S, 60 * scale, rng)
            th = 0.9 + 0.25 * i
            m = smoothstep(th - 0.1, th + 0.1, fld) * 0.9
            img = _layer(img, c, m)

    elif style == "cells":
        # painted patches separated by darker seams (shells, scales, cobbles)
        n = int(spec.get("cells", 40) / max(0.3, scale) ** 2)
        pts = rng.uniform(0, S, (n, 2)).astype(np.float32)
        # tile points for wrap-around
        tiles = np.concatenate([pts + np.array([dx, dy]) * S for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        from scipy.spatial import cKDTree
        gy, gx = np.mgrid[0:S, 0:S].astype(np.float32)
        q = np.stack([gx.ravel(), gy.ravel()], 1)
        wx = smooth_noise(S, S, 40 * scale, rng).ravel() * 12 * scale
        wy = smooth_noise(S, S, 40 * scale, rng).ravel() * 12 * scale
        q = q + np.stack([wx, wy], 1)
        d, idx = cKDTree(tiles).query(q, k=2)
        cell = (idx[:, 0] % n).reshape(S, S)
        seam = (d[:, 1] - d[:, 0]).reshape(S, S)
        pal = cols[1:-1] if len(cols) > 2 else cols
        cell_col = np.array([pal[rng.integers(len(pal))] for _ in range(n)], np.float32)
        img = cell_col[cell]
        # streak shading inside cells
        f = _streak_field(S, angle, 0.6 * scale, rng)
        img *= (1 + 0.09 * f)[..., None]
        seam_col = cols[-1] if len(cols) > 2 else hex2rgb(shade(rgb2hex(cols[0]), -0.3))
        sm = 1 - smoothstep(2.0 * scale, 7.0 * scale, seam)
        img = _layer(img, seam_col, sm * 0.9)

    elif style == "stripes":
        period = float(spec.get("period", 70)) * scale
        gy, gx = np.mgrid[0:S, 0:S].astype(np.float32)
        th = math.radians(angle)
        u = gx * math.cos(th) + gy * math.sin(th)
        u = u + smooth_noise(S, S, 50 * scale, rng) * period * 0.18
        k = len(cols) - 1 if len(cols) > 1 else 1
        band = np.floor(u / period).astype(np.int64) % max(2, k + 1)
        img = base.copy()
        for i, c in enumerate(cols[1:]):
            m = (band == (i + 1)).astype(np.float32)
            m = cv2.GaussianBlur(m, (0, 0), 1.2)
            img = _layer(img, c, m)
        f = _streak_field(S, angle + 90, 0.7 * scale, rng)
        img *= (1 + 0.07 * f)[..., None]

    elif style == "dots":
        img = base.copy()
        f = _streak_field(S, angle, scale, rng)
        img *= (1 + 0.06 * f)[..., None]
        step = float(spec.get("period", 90)) * scale
        r = step * float(spec.get("dot", 0.28))
        mask = np.zeros((S, S), np.float32)
        for yy in np.arange(step / 2, S, step):
            for xx in np.arange(step / 2, S, step):
                cx, cy = xx + rng.uniform(-0.15, 0.15) * step, yy + rng.uniform(-0.15, 0.15) * step
                cv2.ellipse(mask, (int(cx * 4), int(cy * 4)),
                            (int(r * rng.uniform(0.85, 1.15) * 4), int(r * rng.uniform(0.85, 1.15) * 4)),
                            rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA, 2)
        img = _layer(img, cols[1] if len(cols) > 1 else hex2rgb("#222222"), mask)

    elif style == "speckle":
        img = base.copy()
        f = _streak_field(S, angle, scale, rng)
        if len(cols) > 2:
            img = _layer(img, cols[2], smoothstep(0.4, 1.4, f + 0.6 * smooth_noise(S, S, 150, rng)) * 0.7)
        img *= (1 + 0.05 * f)[..., None]
        mask = np.zeros((S, S), np.float32)
        nd = int(S * S / 900 * float(spec.get("density", 1.0)))
        xs, ys = rng.uniform(0, S, nd), rng.uniform(0, S, nd)
        rs = rng.gamma(1.6, 1.3, nd) * scale
        for x, y, r in zip(xs, ys, rs):
            cv2.circle(mask, (int(x * 4), int(y * 4)), max(1, int(r * 4)), 1.0, -1, cv2.LINE_AA, 2)
        img = _layer(img, cols[1] if len(cols) > 1 else hex2rgb("#ffffff"), mask * 0.9)
    else:
        raise ValueError(f"unknown paint style {style!r}")

    # --- bristle marks: fine parallel lines along the stroke direction
    if style not in ("flat",):
        br = aniso_noise(S, angle, 22 * scale, 0.7, rng)
        img *= (1 + 0.055 * contrast * br)[..., None]
    # --- uneven pigment + paper grain
    tone = smooth_noise(S, S, 140, rng)
    img *= (1 + 0.04 * tone)[..., None]
    grain = cv2.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), 0.6)
    img += (0.018 * grain)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- preset paints

PRESETS = {
    # nature
    "sky":        {"colors": ["#6fb7e9", "#9fd3f2", "#3f8fd4", "#c8e8f8"], "style": "streaks", "angle": 5},
    "night":      {"colors": ["#1d2b5c", "#2c3f80", "#141c3f", "#3d4f9a"], "style": "swirl"},
    "dusk":       {"colors": ["#f39a4c", "#f7c25b", "#e2624a", "#c94b6d"], "style": "streaks", "angle": 2},
    "sea":        {"colors": ["#1f7fc4", "#3fa9de", "#155d9c", "#7fd0ec", "#0f9c9a"], "style": "swirl"},
    "deep_sea":   {"colors": ["#0e3f78", "#1a5fa6", "#082a52", "#2b7fbf"], "style": "swirl"},
    "foam":       {"colors": ["#eef8fc", "#cfe9f5", "#ffffff", "#a9d8ee"], "style": "dabs"},
    "grass":      {"colors": ["#4caf3b", "#8fd14f", "#2e7d32", "#c3e35a", "#1b5e20"], "style": "streaks", "angle": 75},
    "leaf":       {"colors": ["#3d9a3a", "#7cc542", "#21652a", "#b8db4c"], "style": "swirl"},
    "dark_leaf":  {"colors": ["#1f5e33", "#2f7f45", "#123f22", "#4f9f4f"], "style": "swirl"},
    "bark":       {"colors": ["#7a4b2a", "#9c6537", "#4e2e18", "#b07a45"], "style": "streaks", "angle": 88},
    "earth":      {"colors": ["#9a6b3c", "#b8834e", "#6e4826", "#cf9f62"], "style": "dabs"},
    "sand":       {"colors": ["#e9c77f", "#f4dc9e", "#d3a95d", "#f8e8bc"], "style": "sponge"},
    "stone":      {"colors": ["#8d8f94", "#a9abb0", "#6b6d72", "#c2c3c6"], "style": "dabs"},
    "mountain":   {"colors": ["#6d6fa8", "#8f8fc4", "#4d4f86", "#a4b2d8"], "style": "streaks", "angle": 60},
    "snow":       {"colors": ["#f6f8fb", "#e1e8f2", "#ffffff", "#c9d6e8"], "style": "dabs"},
    "cloud":      {"colors": ["#ffffff", "#e6f1f9", "#ffffff", "#d2e6f4"], "style": "swirl", "contrast": 0.5},
    "meadow_far": {"colors": ["#84bd4c", "#a9d66a", "#5f9c3c", "#cfe47e", "#4f8a5a"], "style": "streaks", "angle": 60, "contrast": 0.85},
    "sun":        {"colors": ["#ffd23a", "#ffe98a", "#ff9e1f", "#ffb627"], "style": "swirl"},
    "sun_fire":   {"colors": ["#ff8a1f", "#ffd23a", "#e8431c", "#ffb627", "#c7241b"], "style": "swirl"},
    "fire":       {"colors": ["#ff7a1a", "#ffd02e", "#e2361b", "#ffae2a"], "style": "streaks", "angle": 90},
    "moon":       {"colors": ["#f5efc8", "#fffbe2", "#e2d79b", "#ffffff"], "style": "dabs"},
    "water":      {"colors": ["#3b9fd8", "#68c1ea", "#1f78bd", "#a2dcf3"], "style": "streaks", "angle": 0},
    "river":      {"colors": ["#4aa8d8", "#7ccbea", "#2a7fbf", "#c8ecf8"], "style": "streaks", "angle": 0},
    "mud":        {"colors": ["#8a6a3c", "#a5844f", "#5f4524", "#b9985e"], "style": "dabs"},
    "peach_blossom": {"colors": ["#f7a3c0", "#fcd0de", "#e66f9b", "#ffe6ee"], "style": "dabs"},
    "peach_fruit":   {"colors": ["#f7a07a", "#ffd09a", "#e8604f", "#ffc0a8"], "style": "swirl"},
    "flower_red": {"colors": ["#e8392f", "#ff6f4d", "#b81f2a", "#ff9a6b"], "style": "swirl"},
    "flower_yellow": {"colors": ["#ffd43b", "#fff08a", "#f5a524"], "style": "swirl"},
    "flower_purple": {"colors": ["#8e5bd1", "#b38bf0", "#5d34a0", "#d2b5ff"], "style": "swirl"},
    # creatures & people
    "fur_brown":  {"colors": ["#9a6a3f", "#bd8a55", "#6d4422", "#d9ad78"], "style": "streaks"},
    "fur_grey":   {"colors": ["#7d8595", "#9ea6b5", "#555c6b", "#b8c3d6", "#5a6a8a"], "style": "streaks"},
    "fur_white":  {"colors": ["#f7f5ef", "#e6e3dc", "#ffffff", "#d7dbe2"], "style": "streaks"},
    "shell":      {"colors": ["#7c8f3a", "#9db24a", "#a7853d", "#6b7b2e", "#3f4a1c"], "style": "cells", "cells": 28},
    "olive_skin": {"colors": ["#93a24c", "#b3c25e", "#6f7d34", "#c9d27a"], "style": "dabs"},
    "skin_light": {"colors": ["#f6caa0", "#fbdcbc", "#e9ad83", "#ffe7d0"], "style": "flat"},
    "skin_tan":   {"colors": ["#d99a64", "#e8b37f", "#b97a48", "#f0c79a"], "style": "flat"},
    "skin_bronze":{"colors": ["#b9733f", "#d08c52", "#8f532a", "#e2a56c"], "style": "dabs"},
    "red_cloth":  {"colors": ["#d7262e", "#f04a3e", "#a4161f", "#ff6f5e"], "style": "streaks"},
    "blue_cloth": {"colors": ["#2f5fb3", "#4a7fd6", "#1d3f82", "#7aa6ee"], "style": "streaks"},
    "yellow_cloth": {"colors": ["#f2c230", "#ffe070", "#d79a1a"], "style": "streaks"},
    "green_cloth":{"colors": ["#3f8f4f", "#5fb36a", "#2a6a39", "#8fd18f"], "style": "streaks"},
    "plaid":      {"colors": ["#c4372f", "#2b2b2b", "#e2574c"], "style": "stripes", "period": 46, "angle": 0},
    "wicker":     {"colors": ["#c9934d", "#e0b26b", "#9b6a33", "#f0cd8c"], "style": "stripes", "period": 18, "angle": 30},
    "wood":       {"colors": ["#a66b38", "#c4884d", "#7c4c24", "#d9a46a"], "style": "streaks", "angle": 0},
    "hair_black": {"colors": ["#2a2a35", "#40404f", "#16161d", "#55556a"], "style": "streaks", "angle": 80},
    "hair_brown": {"colors": ["#5b3a22", "#7a5030", "#3b2414", "#8f6440"], "style": "streaks", "angle": 80},
    "hair_white": {"colors": ["#f2f2f2", "#dedede", "#ffffff", "#c8c8d0"], "style": "streaks", "angle": 80},
    "feather_black": {"colors": ["#2b2d3a", "#3f4356", "#15161e", "#4b5a7a", "#2d4a5e"], "style": "streaks"},
    "white_beak": {"colors": ["#f5f3ea", "#ffffff", "#dcd8c8"], "style": "flat"},
    "pink":       {"colors": ["#f48fb1", "#f8bbd0", "#e05d8e"], "style": "flat"},
    "black":      {"colors": ["#1d1d22", "#2c2c33", "#101014"], "style": "flat"},
    "white":      {"colors": ["#fbfaf6", "#f2f0ea", "#ffffff"], "style": "flat"},
    "gold":       {"colors": ["#e8b52c", "#ffd866", "#b8841b", "#fff0a8"], "style": "swirl"},
    "brick":      {"colors": ["#b5523b", "#cf6b4f", "#8a3a28", "#d98a6a"], "style": "dabs"},
    "roof":       {"colors": ["#8c3b2e", "#a84b39", "#64261d", "#c06048"], "style": "stripes", "period": 36, "angle": 90},
    "cream":      {"colors": ["#f6ead0", "#fff6e2", "#e5d3ad"], "style": "flat"},
}


def resolve_paint(paint, story_paints: dict | None = None) -> dict:
    """Paint may be a preset name, a hex colour, or a full spec dict."""
    if paint is None:
        paint = "#888888"
    if isinstance(paint, str):
        if story_paints and paint in story_paints:
            return resolve_paint(story_paints[paint], None)
        if paint in PRESETS:
            return dict(PRESETS[paint])
        if paint.startswith("#"):
            return {"colors": [paint], "style": "streaks"}
        raise KeyError(f"unknown paint {paint!r}")
    if isinstance(paint, dict) and "preset" in paint:
        d = dict(PRESETS[paint["preset"]])
        d.update({k: v for k, v in paint.items() if k != "preset"})
        return d
    return dict(paint)


class TextureBank:
    """Memory + disk cache of generated textures."""

    def __init__(self, cache_dir: str | None = None):
        self.mem: dict[str, np.ndarray] = {}
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, spec: dict, size: int = 1024, seed: int = 0) -> np.ndarray:
        key = hashlib.sha1(json.dumps([spec, size, seed], sort_keys=True).encode()).hexdigest()[:16]
        if key in self.mem:
            return self.mem[key]
        path = os.path.join(self.cache_dir, key + ".npy") if self.cache_dir else None
        if path and os.path.exists(path):
            tex = np.load(path).astype(np.float32)
        else:
            tex = paint_texture(spec, size, seed)
            if path:
                tmp = path + f".{os.getpid()}.tmp.npy"
                np.save(tmp, tex.astype(np.float16))
                os.replace(tmp, path)
        self.mem[key] = tex
        return tex
