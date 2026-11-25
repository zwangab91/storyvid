"""Collage renderer: parts -> painted cut-paper sprites -> animated frames.

Coordinates are design units on a 1920x1080 page (x right, y down).
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as G
from .anim import EASE, T, apply_anims, new_state, part_anim_state, state_matrix
from .paper import TextureBank, resolve_paint

DESIGN_W, DESIGN_H = 1920, 1080
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CANDIDATES = {
    "title": [os.path.join(FONT_DIR, "Fredoka-Bold.ttf")],
    "caption": [os.path.join(FONT_DIR, "Andika-Regular.ttf")],
    "caption_bold": [os.path.join(FONT_DIR, "Andika-Bold.ttf")],
    "title_zh": [os.environ.get("STORYVID_FONT_ZH_TITLE", ""),
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
                 "C:/Windows/Fonts/msyhbd.ttc"],
    "caption_zh": [os.environ.get("STORYVID_FONT_ZH", ""),
                   "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/PingFang.ttc",
                   "/System/Library/Fonts/STHeiti Light.ttc", "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/msyh.ttc"],
}
FONTS = {k: next((p for p in v if p and os.path.exists(p)), v[-1]) for k, v in _FONT_CANDIDATES.items()}

EDGE_AMP = {"clean": 0.0, "cut": 1.3, "torn": 3.2}
SHADOW = {"strength": 0.20, "offset": (4.0, 5.0), "blur": 3.5}


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def font_path(name: str) -> str:
    return FONTS.get(name, name)


# --------------------------------------------------------------------------- sprites

class Sprite:
    __slots__ = ("img", "ox", "oy", "rs", "mips")

    def __init__(self, img, ox, oy, rs):
        self.img, self.ox, self.oy, self.rs = img, ox, oy, rs
        self.mips = [img]

    def mip(self, k):
        while len(self.mips) <= k:
            prev = self.mips[-1]
            h, w = prev.shape[:2]
            if h < 8 or w < 8:
                break
            self.mips.append(cv2.resize(prev, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
        return self.mips[min(k, len(self.mips) - 1)], 2 ** min(k, len(self.mips) - 1)


def part_polygon(part: dict):
    return G.shape_polygon(part)


def part_center(part: dict):
    if "c" in part:
        return tuple(part["c"])
    if part.get("shape") == "text":
        return tuple(part.get("c", (0, 0)))
    P = part_polygon(part)
    return tuple(P.mean(0))


_IMG_CACHE: dict = {}


def _load_rgba(path: str) -> np.ndarray:
    """Straight-alpha float32 RGBA (RGB order) of an image file, cached."""
    img = _IMG_CACHE.get(path)
    if img is None:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        _IMG_CACHE[path] = img
    return img


def rasterize_image_part(part: dict, rs: float) -> Sprite:
    """A piece of generated artwork used as a paper part: {"shape": "image", "src": path, "xywh": [x, y, w, h]}.

    The image's own alpha is the cut line; "clip" (polygon) and "cut" (shapes) can trim it further.
    """
    img = _load_rgba(part["src"])
    ih, iw = img.shape[:2]
    x, y, w, h = part.get("xywh", [0, 0, iw, ih])
    sx, sy = w / iw, h / ih
    a = img[..., 3]
    ys, xs = np.where(a > 0.004)
    if len(xs) == 0:
        return Sprite(np.zeros((2, 2, 4), np.float32), x, y, rs)
    shadow_k = float(part.get("shadow", 0.6)) * SHADOW["strength"]
    pad = 4 + (SHADOW["offset"][1] + 3 * SHADOW["blur"] if shadow_k > 0 else 0)
    bx0, by0 = x + xs.min() * sx - pad, y + ys.min() * sy - pad
    bx1, by1 = x + (xs.max() + 1) * sx + pad, y + (ys.max() + 1) * sy + pad
    if "clip" in part:
        P = np.asarray(part["clip"], float)
        bx0, by0 = max(bx0, P[:, 0].min() - pad), max(by0, P[:, 1].min() - pad)
        bx1, by1 = min(bx1, P[:, 0].max() + pad), min(by1, P[:, 1].max() + pad)
    W_ = int(math.ceil((bx1 - bx0) * rs)) + 1
    H_ = int(math.ceil((by1 - by0) * rs)) + 1
    f = sx * rs  # source px -> sprite px
    if f < 0.999:  # shrinking: area-filter once, then place
        src = cv2.resize(img, (max(1, round(iw * f)), max(1, round(ih * f * sy / sx))), interpolation=cv2.INTER_AREA)
        A = np.float32([[iw * f / src.shape[1] * 1.0, 0, (x - bx0) * rs], [0, ih * sy * rs / src.shape[0], (y - by0) * rs]])
        flags = cv2.INTER_LINEAR
    else:
        src = img
        A = np.float32([[sx * rs, 0, (x - bx0) * rs], [0, sy * rs, (y - by0) * rs]])
        flags = cv2.INTER_CUBIC
    rgba = cv2.warpAffine(src, A, (W_, H_), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = np.clip(rgba[..., 3], 0, 1) * float(part.get("opacity", 1.0))
    if "clip" in part or part.get("cut"):
        m8 = np.zeros((H_, W_), np.uint8)
        if "clip" in part:
            cv2.fillPoly(m8, [np.round((np.asarray(part["clip"], float) - [bx0, by0]) * rs * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        else:
            m8[:] = 255
        mask = m8.astype(np.float32) / 255.0
        for cut in part.get("cut", []):
            c8 = np.zeros((H_, W_), np.uint8)
            Q = G.shape_polygon(cut)
            cv2.fillPoly(c8, [np.round((Q - [bx0, by0]) * rs * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            mask *= 1.0 - c8.astype(np.float32) / 255.0
        alpha *= mask
    out = np.zeros((H_, W_, 4), np.float32)
    out[..., :3] = np.clip(rgba[..., :3], 0, 1) * alpha[..., None]
    out[..., 3] = alpha
    if shadow_k > 0:
        dx, dy = SHADOW["offset"]
        sh = cv2.warpAffine(alpha, np.float32([[1, 0, dx * rs], [0, 1, dy * rs]]), (W_, H_))
        sh = cv2.GaussianBlur(sh, (0, 0), SHADOW["blur"] * rs) * shadow_k
        out[..., 3] = alpha + (1 - alpha) * sh
    return Sprite(out, bx0, by0, rs)


def rasterize_part(part: dict, rs: float, bank: TextureBank, story_paints: dict, seed: int) -> Sprite:
    if part.get("shape") == "image":
        return rasterize_image_part(part, rs)
    rng = np.random.default_rng(seed)
    shadow_k = float(part.get("shadow", 1.0)) * SHADOW["strength"]
    edge = part.get("edge", "cut")
    amp = EDGE_AMP.get(edge, edge if isinstance(edge, (int, float)) else 1.3)

    if part.get("shape") == "text":
        mask, ox, oy = _text_mask(part, rs)
        h, w = mask.shape
    else:
        P = part_polygon(part)
        bw, bh = np.ptp(P[:, 0]), np.ptp(P[:, 1])
        amp = min(amp, 0.035 * min(bw, bh) + 0.2)
        if amp > 0:
            P = G.roughen(P, amp, seed=seed, freq=2.0 if edge == "torn" else 1.0)
        pad = 4 + amp + (SHADOW["offset"][1] + 3 * SHADOW["blur"] if shadow_k > 0 else 0)
        ox, oy = P[:, 0].min() - pad, P[:, 1].min() - pad
        w = int(math.ceil((P[:, 0].max() + pad - ox) * rs)) + 1
        h = int(math.ceil((P[:, 1].max() + pad - oy) * rs)) + 1
        m8 = np.zeros((h, w), np.uint8)
        pts = np.round((P - [ox, oy]) * rs * 16).astype(np.int32)
        cv2.fillPoly(m8, [pts], 255, cv2.LINE_AA, shift=4)
        mask = m8.astype(np.float32) / 255.0
        for cut in part.get("cut", []):  # holes / bites / crescents
            c8 = np.zeros((h, w), np.uint8)
            Q = G.shape_polygon(cut)
            cv2.fillPoly(c8, [np.round((Q - [ox, oy]) * rs * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            mask *= 1.0 - c8.astype(np.float32) / 255.0

    # --- paint
    spec = resolve_paint(part.get("paint", "#888888"), story_paints)
    tex_size = 1024 if max(w, h) <= 1100 else 2048
    variant = int(part.get("tex_variant", seed % 3))
    tex = bank.get(spec, tex_size, variant + (int(_h(spec), 16) % 997))
    th = math.radians(float(part.get("tex_rot", rng.uniform(-25, 25))))
    ts = float(part.get("tex_scale", 1.0)) * rs
    tx, ty = rng.uniform(0, tex_size * 0.6, 2)
    A = np.array([[ts * math.cos(th), -ts * math.sin(th), 0], [ts * math.sin(th), ts * math.cos(th), 0]])
    A[:, 2] = -(A[:, :2] @ [tx, ty])
    rgb = cv2.warpAffine(tex, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = mask * float(part.get("opacity", 1.0))
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = rgb * alpha[..., None]
    out[..., 3] = alpha
    if shadow_k > 0:
        dx, dy = SHADOW["offset"]
        Ms = np.float32([[1, 0, dx * rs], [0, 1, dy * rs]])
        sh = cv2.warpAffine(mask, Ms, (w, h))
        sh = cv2.GaussianBlur(sh, (0, 0), SHADOW["blur"] * rs) * shadow_k
        out[..., 3] = alpha + (1 - alpha) * sh  # paper over its own soft shadow (shadow colour = black)
    return Sprite(out, ox, oy, rs)


def _text_mask(part, rs):
    size = float(part.get("size", 100))
    font = ImageFont.truetype(font_path(part.get("font", "title")), max(4, int(size * rs)))
    text = part["text"]
    l, t, r, b = font.getbbox(text)
    pad = int(20 * rs)
    W, H = r - l + 2 * pad, b - t + 2 * pad
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    sw = int(part.get("stroke", 0) * rs)
    d.text((pad - l, pad - t), text, font=font, fill=255, stroke_width=sw, stroke_fill=255)
    mask = np.asarray(im, np.float32) / 255.0
    cx, cy = part.get("c", (0, 0))
    ox = cx - (W / 2) / rs
    oy = cy - (H / 2) / rs
    return mask, ox, oy


# --------------------------------------------------------------------------- actors

class Actor:
    """A named set of parts with optional parent links and rigs (part-level anims)."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.parts = spec["parts"]
        self.ids = {p.get("id", f"_{i}"): i for i, p in enumerate(self.parts)}
        self.pivots = []
        for p in self.parts:
            pv = p.get("pivot")
            self.pivots.append(tuple(pv) if pv is not None else part_center(p))
        self.parent_idx = [self.ids.get(p.get("parent")) if p.get("parent") else None for p in self.parts]
        self.rigs = spec.get("rigs", {})
        self.key = name if name and not name.startswith("_inline") else _h(spec)

    def rig_anims(self, rig_names, extra=None):
        """-> {part_index: [anims]}"""
        out: dict[int, list] = {}
        names = rig_names if isinstance(rig_names, list) else ([rig_names] if rig_names else [])
        entries = []
        for n in names:
            if isinstance(n, dict):  # {"name": "run", "t": [a, b]} -> window every entry
                for e in self.rigs.get(n["name"], []):
                    e = dict(e)
                    if "t" in n and "t" not in e:
                        e["t"] = n["t"]
                    entries.append(e)
            else:
                if n not in self.rigs:
                    raise KeyError(f"actor {self.name!r} has no rig {n!r} (has {list(self.rigs)})")
                entries += self.rigs[n]
        entries += extra or []
        for e in entries:
            targets = e["part"] if isinstance(e["part"], list) else [e["part"]]
            for t in targets:
                if t not in self.ids:
                    raise KeyError(f"actor {self.name!r} has no part {t!r}")
                out.setdefault(self.ids[t], []).append(e)
        return out


# --------------------------------------------------------------------------- frame compositing

def blit(dst: np.ndarray, spr: Sprite, M: np.ndarray, opacity: float = 1.0):
    """Composite premultiplied sprite onto dst (H,W,3 float) with 3x3 matrix M (sprite px -> screen px)."""
    s = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    k = 0 if s >= 0.71 else int(math.floor(math.log2(1 / s) + 0.5))
    img, f = spr.mip(k)
    if f != 1:
        M = M @ np.diag([f, f, 1.0])
    h, w = img.shape[:2]
    cs = M @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], np.float64)
    H, W = dst.shape[:2]
    x0, x1 = max(0, int(math.floor(cs[0].min()))), min(W, int(math.ceil(cs[0].max())) + 1)
    y0, y1 = max(0, int(math.floor(cs[1].min()))), min(H, int(math.ceil(cs[1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M[:2].copy()
    M2[:, 2] -= [x0, y0]
    wimg = cv2.warpAffine(img, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if opacity < 0.999:
        wimg *= opacity
    roi = dst[y0:y1, x0:x1]
    roi *= 1.0 - wimg[..., 3:4]
    roi += wimg[..., :3]


def paper_background(w, h, color="#fbf9f3", seed=1):
    from .paper import hex2rgb, smooth_noise
    rng = np.random.default_rng(seed)
    base = np.ones((h, w, 3), np.float32) * hex2rgb(color)
    tone = smooth_noise(h, w, 200 * w / 1920, rng)
    grain = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.7)
    base *= (1 + 0.012 * tone + 0.012 * grain)[..., None]
    return np.clip(base, 0, 1)


class SceneRenderer:
    """Renders frames of one storyboard scene (already normalised by pipeline.build_scenes)."""

    def __init__(self, story: dict, actors: dict[str, Actor], out_w: int, out_h: int,
                 bank: TextureBank, sprite_cache: dict):
        self.story = story
        self.actors = actors
        self.W, self.H = out_w, out_h
        self.k = out_w / DESIGN_W
        self.bank = bank
        self.cache = sprite_cache
        self.paints = story.get("paints", {})
        self.cur_scene = 0
        self.last_use: dict = {}

    # ---- helpers
    def _actor_for(self, item) -> Actor:
        if "actor" in item:
            name = item["actor"]
            if name not in self.actors:
                raise KeyError(f"unknown actor {name!r}")
            return self.actors[name]
        key = "_inline_" + _h({k: item[k] for k in ("parts",) if k in item} or item.get("part"))
        if key not in self.actors:
            parts = item.get("parts") or [item["part"]]
            self.actors[key] = Actor(key, {"parts": parts, "rigs": item.get("rigs", {})})
        return self.actors[key]

    def _sprite(self, actor: Actor, idx: int, rs: float, seed: int) -> Sprite:
        key = (actor.key, idx, rs, seed)
        self.last_use[key] = self.cur_scene
        spr = self.cache.get(key)
        if spr is None:
            part = actor.parts[idx]
            if seed and part.get("shape") in ("wave", "blob", "cloud") and not part.get("fixed_shape"):
                part = dict(part, shape_seed=part.get("shape_seed", 0) + seed, phase=part.get("phase", 0.0) + seed * 1.7)
            spr = rasterize_part(part, rs, self.bank, self.paints,
                                 seed * 1009 + idx * 31 + int(actor.key.encode().hex()[:6], 16) % 10007)
            self.cache[key] = spr
        return spr

    def camera(self, scene, u):
        cam = scene.get("camera") or {}
        z = cam.get("zoom", [1.0, 1.035])
        z = (z, z) if np.isscalar(z) else z
        c = cam.get("center", [[960, 540], [960, 540]])
        if np.isscalar(c[0]):
            c = [c, c]
        p = EASE.get(cam.get("ease", "inout"), EASE["inout"])(u)
        zz = z[0] + (z[1] - z[0]) * p
        cx = c[0][0] + (c[1][0] - c[0][0]) * p
        cy = c[0][1] + (c[1][1] - c[0][1]) * p
        M = np.diag([self.k, self.k, 1.0]) @ T(DESIGN_W / 2, DESIGN_H / 2) @ np.diag([zz, zz, 1.0]) @ T(-cx, -cy)
        if cam.get("shake"):
            sh = cam["shake"]
            t0, t1 = sh.get("t", (0, 1))
            if t0 <= u <= t1:
                rng = np.random.default_rng(int(u * scene["_dur"] * 12) + 99)
                M = T(*(rng.uniform(-1, 1, 2) * sh.get("amp", 6) * self.k)) @ M
        return M, max(z)

    def raster_scale(self, item, zmax):
        smax = item.get("scale", 1.0)
        smax = max(smax) if not np.isscalar(smax) else smax
        mult = 1.0
        for a in item.get("anims", []):
            if a.get("type") == "scale":
                to = a["to"]
                mult = max(mult, max(to) / smax if not np.isscalar(to) else to / max(smax, 1e-6))
            elif a.get("type") == "pulse":
                mult = max(mult, 1 + a.get("amp", 0.05))
            elif a.get("type") == "pop":
                mult = max(mult, 1.12)
        need = self.k * zmax * abs(smax) * mult * float(item.get("detail", 1.0))
        return max(0.25, 2 ** (math.ceil(math.log2(max(need, 1e-3)) * 4 - 1e-6) / 4))  # 2^(1/4) steps

    # ---- main entry
    def render(self, scene: dict, u: float, dst: np.ndarray):
        dur = scene["_dur"]
        C, zmax = self.camera(scene, u)
        for item in scene["_items"]:
            actor = self._actor_for(item)
            seed = int(item.get("seed", 0))
            if "_origin" not in item:
                if "at" not in item and "actor" not in item:
                    # inline part with absolute coords: pivot item anims about the part's centre
                    c = part_center(actor.parts[0]) if len(actor.parts) == 1 else (0.0, 0.0)
                    item["_origin"] = tuple(map(float, c))
                    item["at"] = list(c)
                else:
                    item["_origin"] = tuple(item.get("origin", (0.0, 0.0)))
            st = new_state(*item.get("at", (0, 0)), scale=item.get("scale", 1.0), rot=item.get("rot", 0.0),
                           flip=item.get("flip", False), opacity=item.get("opacity", 1.0))
            st = apply_anims(st, item.get("anims", []), u, dur, seed)
            if not st["vis"] or st["op"] <= 0.003 or abs(st["sx"]) < 1e-3 or abs(st["sy"]) < 1e-3:
                continue
            I = C @ state_matrix(st) @ T(-item["_origin"][0], -item["_origin"][1])
            rs = item.get("_rs") or self.raster_scale(item, zmax)
            item["_rs"] = rs
            rig = item.get("_rig_cache")
            if rig is None:
                rig = actor.rig_anims(item.get("rig"), item.get("part_anims"))
                item["_rig_cache"] = rig
            hidden = set(item.get("hide", []))
            reveal = set(item.get("reveal", []))
            n = len(actor.parts)
            local, popv, visv = [None] * n, [1.0] * n, [True] * n
            for i in range(n):
                pv = actor.pivots[i]
                pa = rig.get(i)
                if pa:
                    ps = part_anim_state(pa, u, dur, seed + i)
                    local[i] = T(pv[0] + ps["x"], pv[1] + ps["y"]) @ state_matrix(
                        {**ps, "x": 0.0, "y": 0.0}) @ T(-pv[0], -pv[1])
                    popv[i], visv[i] = ps["op"], ps["vis"]
                else:
                    local[i] = None
            world = [None] * n

            def W_of(i, depth=0):
                if world[i] is None:
                    par = actor.parent_idx[i]
                    base = W_of(par, depth + 1) if (par is not None and depth < 32) else np.eye(3)
                    world[i] = base @ local[i] if local[i] is not None else base
                return world[i]
            for i, part in enumerate(actor.parts):
                pid = part.get("id", f"_{i}")
                Wm = W_of(i)
                pop, pvis = popv[i], visv[i]
                if not pvis or pid in hidden or (part.get("hidden") and pid not in reveal):
                    continue
                spr = self._sprite(actor, i, rs, seed)
                K = np.array([[1 / spr.rs, 0, spr.ox], [0, 1 / spr.rs, spr.oy], [0, 0, 1]])
                blit(dst, spr, I @ Wm @ K, st["op"] * pop)
