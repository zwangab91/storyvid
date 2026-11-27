"""storyvid pipeline: storyboard JSON -> narrated painted-paper collage video.

    python -m storyvid.pipeline render stories/hare.json --out out/hare
    python -m storyvid.pipeline render stories/hare.json --out out/hare --preview   # contact sheet only
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from . import audio as A
from .library import LIBRARY, expand_template
from .paper import TextureBank
from .render import (DESIGN_H, DESIGN_W, FONTS, Actor, SceneRenderer, blit, paper_background, Sprite)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("STORYVID_CACHE", os.path.join(ROOT, ".cache"))

XFADE = 0.6       # cross-dissolve between scenes (s)
LEAD = 0.75       # silence before a scene's narration (s)
TAIL = 0.95       # silence after it (s)
TITLE_PALETTE = ["red_cloth", "sun", "grass", "sky", "flower_purple", "fire", "sea", "peach_fruit", "leaf", "gold"]


# --------------------------------------------------------------------------- story normalisation

def load_story(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        story = json.load(f)
    base = os.path.dirname(os.path.abspath(path))

    def fix(parts):  # image parts: paths are relative to the storyboard file
        for p in parts or []:
            if p.get("shape") == "image" and not os.path.isabs(p["src"]):
                p["src"] = os.path.join(base, p["src"])
    for a in (story.get("actors") or {}).values():
        fix(a.get("parts"))
    for sc in story.get("scenes", []) + [story.get("title_scene") or {}, story.get("end_scene") or {}]:
        for it in sc.get("items", []):
            fix(it.get("parts"))
            if it.get("part"):
                fix([it["part"]])
    if story.get("soundtrack") and not os.path.isabs(story["soundtrack"]):  # folder made by storyvid.soundtrack
        story["soundtrack"] = os.path.join(base, story["soundtrack"])
    return story


def build_actors(story: dict) -> dict[str, Actor]:
    actors = {k: Actor(k, v) for k, v in LIBRARY.items()}
    for k, v in (story.get("actors") or {}).items():
        actors[k] = Actor(k, expand_template(copy.deepcopy(v)))
    return actors


def _expand_items(items: list) -> list:
    out = []
    for it in items:
        sc = it.get("scatter")
        if not sc:
            out.append(it)
            continue
        rng = np.random.default_rng(sc.get("seed", 0))
        x0, y0, x1, y1 = sc["area"]
        smin, smax = sc.get("scale", [0.8, 1.0])
        pts = []
        for i in range(sc.get("n", 8)):
            for _ in range(20):  # simple blue-noise: avoid clumping
                p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > sc.get("min_dist", 40) ** 2 for q in pts):
                    break
            pts.append(p)
        for i, p in enumerate(sorted(pts, key=lambda q: q[1])):
            c = {k: v for k, v in it.items() if k != "scatter"}
            c["at"] = [p[0], p[1]]
            c["scale"] = float(rng.uniform(smin, smax)) * (it.get("scale", 1.0) if np.isscalar(it.get("scale", 1.0)) else 1)
            c["seed"] = int(it.get("seed", 0)) + i + 1
            if sc.get("flip_random"):
                c["flip"] = bool(rng.random() < 0.5)
            # de-synchronise looping anims
            c["anims"] = [dict(a, phase=a.get("phase", 0) + rng.uniform(0, 1)) if "period" in a else a
                          for a in it.get("anims", [])]
            out.append(c)
    return out


def normalise_scene(scene: dict, index: int = 0) -> dict:
    sc = copy.deepcopy(scene)
    for k, it in enumerate(sc.get("items", [])):
        if it.get("actor") in LIBRARY and "seed" not in it:
            it["seed"] = (index * 7 + k * 3) % 41 + 1
    items = []
    bg = sc.get("background")
    if bg and bg not in ("paper", "white"):
        cam = sc.get("camera") or {}
        z = cam.get("zoom", [1.0, 1.035])
        zmax = max(z) if not np.isscalar(z) else z
        items.append({"part": {"shape": "rect", "xywh": [-160, -120, 2240, 1320], "paint": bg,
                               "edge": "clean", "shadow": 0}, "detail": 1.0 / zmax, "z": -1000})
    items += _expand_items(sc.get("items", []))
    for i, it in enumerate(items):
        it["_order"] = i
    sc["_items"] = sorted(items, key=lambda it: (it.get("z", 0), it["_order"]))
    return sc


def _title_items(text: str, lang: str, y: float, size: float, seed: int, delay: float = 0.04, span: float = 0.4,
                 paints=None):
    """Painted cut-paper letters, one part per glyph, popping in one by one. `paints` (a paint, or a list cycled
    letter by letter) defaults to the bright TITLE_PALETTE, made for a title on plain paper."""
    paints = TITLE_PALETTE if paints is None else (paints if isinstance(paints, list) else [paints])
    font_key = "title_zh" if lang.startswith("zh") else "title"
    font = ImageFont.truetype(FONTS[font_key], int(size))
    # wrap
    zh = lang.startswith("zh")
    words = list(text) if zh else text.split(" ")
    sep = "" if zh else " "
    total = font.getlength(text)
    nlines = max(1, math.ceil(total / 1600))
    target = total / nlines
    lines, cur = [], ""
    for w in words:
        cand = (cur + sep + w) if cur else w
        if cur and font.getlength(cand) > target * 1.08 and len(lines) < nlines - 1:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    lines.append(cur)
    items, gi = [], 0
    nglyph = sum(1 for ch in text if ch.strip())
    delay = min(delay, span / max(1, nglyph))
    lh = size * 1.15
    y0 = y - (len(lines) - 1) * lh / 2
    rng = np.random.default_rng(seed)
    for li, line in enumerate(lines):
        W = font.getlength(line)
        x = 960 - W / 2
        for j, ch in enumerate(line):
            adv = font.getlength(line[: j + 1]) - font.getlength(line[:j])
            if ch.strip():
                cx = x + font.getlength(line[:j]) + adv / 2
                t0 = 0.05 + delay * gi
                items.append({
                    "part": {"shape": "text", "text": ch, "font": font_key, "size": size, "c": [0, 0],
                             "paint": paints[(gi + seed) % len(paints)], "stroke": 2},
                    "at": [cx, y0 + li * lh], "rot": float(rng.uniform(-6, 6)), "z": 50,
                    "anims": [{"type": "pop", "t": [t0, t0 + 0.12]},
                              {"type": "sway", "deg": 3, "period": 2.5 + rng.uniform(0, 1), "phase": rng.uniform(0, 1)}]})
                gi += 1
    return items


def _small_text(text, lang, y, size=40, color="#3a3a3a", t0=0.3):
    return {"part": {"shape": "text", "text": text, "font": "caption_zh" if lang.startswith("zh") else "caption_bold",
                     "size": size, "c": [0, 0], "paint": {"colors": [color], "style": "flat"}, "shadow": 0},
            "at": [960, y], "z": 60, "anims": [{"type": "fade", "to": 1.0, "t": [t0, t0 + 0.15]}], "opacity": 0.0}


def _plate(y0: float, y1: float, width: float, paint, seed: int) -> dict:
    """A torn strip of paper behind title text, so it reads on a busy painted backdrop."""
    return {"part": {"shape": "rect", "c": [960, (y0 + y1) / 2], "size": [width, y1 - y0], "paint": paint, "edge": "torn"},
            "z": 40, "seed": seed, "anims": [{"type": "fade", "to": 1.0, "t": [0.0, 0.08]}], "opacity": 0.0}


def build_scenes(story: dict) -> list[dict]:
    """Title card, story scenes, end card. `title_style` styles the title and end-card text:
    {"paints": paint or [paints] for the letters (default: the bright title palette), "text_color": "#hex" for the
    subtitle and credit lines, "plate": a paint for a torn paper strip behind the text, or omitted for none}."""
    lang = story.get("language", "en")
    zh = lang.startswith("zh")
    ts_style = story.get("title_style") or {}
    letter_paints = ts_style.get("paints")
    text_color = ts_style.get("text_color")
    plate = ts_style.get("plate")
    scenes = []
    # ---- title card (optional)
    ts = story.get("title_scene") or {}
    if story.get("title_card", True) is not False:
        has_art = bool(ts.get("items"))
        ty = 330 if has_art else 470
        tsize = 150 if not zh else 170
        title_items = _title_items(story["title"], lang, ty, tsize, seed=3, paints=letter_paints)
        if story.get("subtitle"):
            title_items.append(_small_text(story["subtitle"], "en" if zh else lang, ty + 140, 44, t0=0.35,
                                           **({"color": text_color} if text_color else {})))
        if plate:
            width = ImageFont.truetype(FONTS["title_zh" if zh else "title"], tsize).getlength(story["title"]) + 220
            title_items.append(_plate(ty - tsize * 0.75, ty + (180 if story.get("subtitle") else tsize * 0.75),
                                      max(width, 760), plate, seed=5))
        scenes.append({"kind": "title", "narration": story.get("title_narration", story["title"]),
                       "background": ts.get("background", "paper"), "camera": ts.get("camera", {"zoom": [1.0, 1.03]}),
                       "items": title_items + ts.get("items", []), "min_duration": 4.5, "captions": False})
    # ---- story
    for s in story["scenes"]:
        scenes.append(dict(s, kind="story"))
    # ---- end card (optional)
    if story.get("end_card", True) is not False:
        es = story.get("end_scene") or {}
        end_text = story.get("end_text", "完" if zh else "The End")
        ey = 420 if es.get("items") else 500
        end_items = _title_items(end_text, lang, ey, 220 if zh else 160, seed=0, delay=0.05, paints=letter_paints)
        if story.get("credit"):
            end_items.append(_small_text(story["credit"], lang, ey + 130, 36, color=text_color or "#6a6a6a", t0=0.25))
        if plate:
            cw = ImageFont.truetype(FONTS["caption_zh" if zh else "caption_bold"], 36).getlength(story.get("credit", ""))
            end_items.append(_plate(ey - 150, ey + (175 if story.get("credit") else 120), max(cw + 160, 600), plate,
                                    seed=6))
        scenes.append({"kind": "end", "narration": es.get("narration", ""), "background": es.get("background", "paper"),
                       "camera": es.get("camera", {"zoom": [1.0, 1.02]}), "items": end_items + es.get("items", []),
                       "min_duration": 4.0, "captions": False})
    return [normalise_scene(s, i) for i, s in enumerate(scenes)]


# --------------------------------------------------------------------------- audio + timeline

def build_audio_timeline(story: dict, scenes: list[dict], cache_dir: str):
    lang = story.get("language", "en")
    voice = story.get("voice", "melo" if lang.startswith("zh") else "af_heart")
    speed = float(story.get("speed", 0.9 if lang.startswith("zh") else 0.92))
    nar = A.Narrator(os.path.join(cache_dir, "tts"))
    clipnar = A.ClipNarrator(story["soundtrack"]) if story.get("soundtrack") else None
    if clipnar:  # generated narration (storyvid.soundtrack): every voiced scene must have its clip
        missing = clipnar.missing([sc.get("narration", "") for sc in scenes])
        if missing:
            raise SystemExit(f"soundtrack {story['soundtrack']} has no clip for {len(missing)} scene(s), e.g. "
                             f"{missing[0][:40]!r}; re-run `storyvid.soundtrack plan` and generate the new lines")
    t = 0.0
    clips = []
    for i, sc in enumerate(scenes):
        text = sc.get("narration", "").strip()
        if text and clipnar:
            a, spans = clipnar.narrate(text, lang)
        elif text:
            a, spans = nar.narrate(text, lang, sc.get("voice", voice), float(sc.get("speed", speed)))
        else:
            a, spans = np.zeros(1, np.float32), []
        nlen = len(a) / A.SR if text else 0.0
        lead = float(sc.get("lead", LEAD))
        dur = lead + nlen + float(sc.get("hold", 0.0)) + TAIL
        dur = max(dur, float(sc.get("min_duration", 3.2)), float(sc.get("duration", 0)))
        sc["_start"], sc["_dur"], sc["_nar_start"] = t, dur, lead
        sc["_spans"] = spans
        if text and clipnar:
            sc["_audio"] = a  # a generated clip pauses at commas too: captions within a sentence follow its pauses
        clips.append((t + lead, a))
        t += dur - (XFADE if i < len(scenes) - 1 else 0)
    total = t
    narration = np.zeros(int((total + 0.5) * A.SR), np.float32)
    for st, a in clips:
        s = int(st * A.SR)
        narration[s:s + len(a)] += a[: len(narration) - s]
    m = story.get("music", {})
    if m is not False and clipnar and clipnar.music and not m.get("procedural"):
        # generated music, balanced as Paperwing plays it: the voice at full level, the music at `volume` (0.28) of
        # its own level, times `duck` (0.45) while a scene is narrated. Our narration is levelled, so scale the music
        # by the same factor the levelling applied to the voice.
        k = 0.085 / float(np.median(clipnar.raw_rms)) if clipnar.raw_rms else 1.0
        vol, duck = float(m.get("volume", 0.28)), float(m.get("duck", 0.45))
        bed = A.loop_music(A.load_music(clipnar.music), total + 0.5)
        voiced = [(sc["_start"], sc["_start"] + sc["_dur"]) for sc in scenes if sc.get("narration", "").strip()]
        return A.mix_bed(narration, bed, voiced, vol * k, vol * duck * k), total
    music = None
    if m is not False:
        style = m.get("style", "pluck" if lang.startswith("zh") else "musicbox")
        music = A.synth_music(total + 0.5, seed=int(m.get("seed", 1)), style=style,
                              tempo=float(m.get("tempo", 70 if style == "pluck" else 84)))
    mix = A.mix_tracks(narration, music, music_db=float(m.get("db", -21)) if m else -21)
    return mix, total


def build_captions(story: dict, scenes: list[dict]):
    lang = story.get("language", "en")
    maxlen = 24 if lang.startswith("zh") else 64
    for sc in scenes:
        caps = []
        if sc.get("captions", True) and sc.get("_spans"):
            subs = sc.get("subtitle_en")
            sub_list = A.split_sentences(subs, "en") if subs else []
            for k, (a, b, sent) in enumerate(sc["_spans"]):
                chunks = A.split_caption(sent, lang, maxlen)
                # keep the sentence's translation on screen for every chunk of that sentence
                sub = sub_list[k] if (k < len(sub_list) and len(sub_list) == len(sc["_spans"])) else None
                if len(chunks) > 1 and sc.get("_audio") is not None:
                    seg = sc["_audio"][int(a * A.SR):int(b * A.SR)]
                    for c0, c1, c in A.align_sentences(seg, chunks, lang):
                        caps.append([sc["_nar_start"] + a + c0, sc["_nar_start"] + a + c1, c, sub])
                    continue
                L = sum(len(c) for c in chunks)
                t = a
                for j, c in enumerate(chunks):
                    d = (b - a) * len(c) / L
                    caps.append([sc["_nar_start"] + t, sc["_nar_start"] + t + d, c, sub])
                    t += d
            if subs and len(sub_list) != len(sc["_spans"]) and caps:
                caps[0][3] = subs  # fallback: show whole translation with the first chunk
            # chain: keep each caption up until the next one starts (no flicker between sentences)
            for j in range(len(caps) - 1):
                caps[j][1] = caps[j + 1][0]
            # linger a little, but be gone before the cross-dissolve into the next scene
            caps[-1][1] = min(caps[-1][1] + 0.5, sc["_dur"] - XFADE - 0.2)
        sc["_captions"] = caps


def write_srt(scenes, path):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    n = 1
    with open(path, "w", encoding="utf-8") as f:
        for sc in scenes:
            for a, b, txt, sub in sc.get("_captions", []):
                f.write(f"{n}\n{ts(sc['_start'] + a)} --> {ts(sc['_start'] + b)}\n{txt}\n" + (f"{sub}\n" if sub else "") + "\n")
                n += 1


# --------------------------------------------------------------------------- caption sprites

def caption_sprite(text: str, sub: str | None, lang: str, k: float) -> Sprite:
    zh = lang.startswith("zh")
    fs = (50 if zh else 46) * k
    font = ImageFont.truetype(FONTS["caption_zh" if zh else "caption"], int(fs))
    sfont = ImageFont.truetype(FONTS["caption"], int(30 * k)) if sub else None
    tw = font.getlength(text)
    sw = sfont.getlength(sub) if sub else 0
    if sub and sw > 1700 * k:  # wrap long translation into two lines
        words, l1 = sub.split(), ""
        for w in words:
            if sfont.getlength((l1 + " " + w).strip()) > max(sw / 2, 600 * k) and l1:
                break
            l1 = (l1 + " " + w).strip()
        sub = l1 + "\n" + sub[len(l1):].strip()
        sw = max(sfont.getlength(x) for x in sub.split("\n"))
    padx, pady = 34 * k, 16 * k
    th = fs * 1.25
    sh = (32 * k * 1.3 * (sub.count("\n") + 1) + 6 * k) if sub else 0
    W = int(max(tw, sw) + 2 * padx)
    H = int(th + sh + 2 * pady)
    m = int(14 * k)
    im = Image.new("RGBA", (W + 2 * m, H + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # soft shadow
    sh_im = Image.new("L", im.size, 0)
    ImageDraw.Draw(sh_im).rounded_rectangle([m + 3 * k, m + 5 * k, m + W + 3 * k, m + H + 5 * k], radius=int(18 * k), fill=70)
    sh_arr = cv2.GaussianBlur(np.asarray(sh_im, np.float32) / 255.0, (0, 0), 5 * k)
    d.rounded_rectangle([m, m, m + W, m + H], radius=int(18 * k), fill=(255, 251, 242, 236))
    d.text((m + W / 2, m + pady + th / 2), text, font=font, fill=(38, 34, 30, 255), anchor="mm")
    if sub:
        d.multiline_text((m + W / 2, m + pady + th + sh / 2), sub, font=sfont, fill=(95, 90, 84, 255),
                         anchor="mm", align="center", spacing=int(4 * k))
    arr = np.asarray(im, np.float32) / 255.0
    a = arr[..., 3]
    a_tot = a + (1 - a) * sh_arr
    out = np.zeros_like(arr)
    out[..., :3] = arr[..., :3] * a[..., None]
    out[..., 3] = a_tot
    spr = Sprite(out, 0, 0, 1.0)
    return spr


# --------------------------------------------------------------------------- frame rendering

class FrameMaker:
    def __init__(self, story, scenes, W, H, cache_dir):
        self.story, self.scenes, self.W, self.H = story, scenes, W, H
        self.k = W / DESIGN_W
        self.lang = story.get("language", "en")
        bank = TextureBank(os.path.join(cache_dir, "textures"))
        self.sprite_cache: dict = {}
        self.renderer = SceneRenderer(story, build_actors(story), W, H, bank, self.sprite_cache)
        self.paper = paper_background(W, H, story.get("paper", "#fbf9f3"))
        self.cap_cache: dict = {}
        self._cache_scene = -1

    def scene_frame(self, i: int, t_local: float) -> np.ndarray:
        sc = self.scenes[i]
        self.renderer.cur_scene = i
        u = min(1.0, max(0.0, t_local / sc["_dur"]))
        f = self.paper.copy()
        self.renderer.render(sc, u, f)
        caps = sc.get("_captions", [])
        for j, (a, b, txt, sub) in enumerate(caps):
            # fade only at the start/end of a run of captions; consecutive captions hard-cut
            fin = 0.15 if (j == 0 or caps[j - 1][1] < a - 0.05) else 0.0
            fout = 0.15 if (j == len(caps) - 1 or caps[j + 1][0] > b + 0.05) else 0.0
            if a - fin <= t_local < b + fout:
                op = 1.0
                if fin and t_local < a:
                    op = (t_local - (a - fin)) / fin
                if fout and t_local > b:
                    op = ((b + fout) - t_local) / fout
                key = (i, j)
                spr = self.cap_cache.get(key)
                if spr is None:
                    spr = self.cap_cache[key] = caption_sprite(txt, sub, self.lang, self.k)
                h, w = spr.img.shape[:2]
                M = np.array([[1, 0, self.W / 2 - w / 2], [0, 1, self.H - h - 26 * self.k], [0, 0, 1]], float)
                blit(f, spr, M, max(0.0, min(1.0, op)))
        return f

    def _evict(self, first_active: int):
        """Free sprites last used by scenes before `first_active` (memory stays ~2 scenes)."""
        if first_active <= self._cache_scene:
            return
        self._cache_scene = first_active
        lu = self.renderer.last_use
        for key in [k for k, v in lu.items() if v < first_active]:
            self.sprite_cache.pop(key, None)
            lu.pop(key, None)

    def frame_at(self, t: float) -> np.ndarray:
        act = [i for i, sc in enumerate(self.scenes) if sc["_start"] <= t < sc["_start"] + sc["_dur"]]
        if not act:
            act = [len(self.scenes) - 1]
        self._evict(act[0])
        if len(act) == 1:
            i = act[0]
            return self.scene_frame(i, t - self.scenes[i]["_start"])
        i, j = act[0], act[-1]
        fa = self.scene_frame(i, t - self.scenes[i]["_start"])
        fb = self.scene_frame(j, t - self.scenes[j]["_start"])
        p = (t - self.scenes[j]["_start"]) / XFADE
        p = p * p * (3 - 2 * p)
        return fa * (1 - p) + fb * p


_CTX: dict = {}


def _init_worker(ctx):
    _CTX.update(ctx)


def _worker(job):
    fm = FrameMaker(_CTX["story"], _CTX["scenes"], _CTX["W"], _CTX["H"], _CTX["cache"])
    f0, f1, path, fps = job
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fm.W}x{fm.H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
           "-threads", "1", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in range(f0, f1):
        fr = fm.frame_at(fi / fps)
        p.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        if (fi - f0) % 240 == 0:
            print(f"  [{os.getpid()}] frame {fi}/{f1} ({(time.time() - t0) / max(1, fi - f0 + 1):.3f}s/f)", flush=True)
    p.stdin.close()
    p.wait()
    return path


def contact_sheet(fm: FrameMaker, scenes, path, cols=4, u=0.6, tw=480):
    tiles = []
    for i, sc in enumerate(scenes):
        f = fm.scene_frame(i, sc["_dur"] * u)
        im = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        im = cv2.resize(im, (tw, int(tw * fm.H / fm.W)), interpolation=cv2.INTER_AREA)
        cv2.putText(im, str(i), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def render(story_path: str, out_dir: str, height: int = 1080, fps: int = 24, workers: int | None = None,
           preview: bool = False, stills: list[float] | None = None, only: list[int] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    story = load_story(story_path)
    name = os.path.splitext(os.path.basename(story_path))[0]
    W, H = int(round(height * 16 / 9 / 2) * 2), height
    t0 = time.time()
    scenes = build_scenes(story)
    mix, total = build_audio_timeline(story, scenes, CACHE)
    build_captions(story, scenes)
    for sc in scenes:
        sc.pop("_audio", None)  # only needed for caption timing; keep it out of the render workers
    print(f"[{name}] {len(scenes)} scenes, {total:.1f}s, audio ready in {time.time() - t0:.1f}s", flush=True)
    with open(os.path.join(out_dir, f"{name}.timeline.json"), "w", encoding="utf-8") as f:
        json.dump([{"i": i, "start": round(s["_start"], 2), "dur": round(s["_dur"], 2),
                    "narration": s.get("narration", "")} for i, s in enumerate(scenes)], f, ensure_ascii=False, indent=1)
    if preview or stills:
        fm = FrameMaker(story, scenes, W, H, CACHE)
        contact_sheet(fm, scenes, os.path.join(out_dir, f"{name}.contact.jpg"))
        for i, sc in enumerate(scenes):
            if only and i not in only:
                continue
            for u in stills or []:
                f = fm.scene_frame(i, sc["_dur"] * u)
                cv2.imwrite(os.path.join(out_dir, f"{name}.s{i:02d}_{int(u * 100):02d}.jpg"),
                            cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[{name}] preview written in {time.time() - t0:.1f}s")
        return None
    wav = os.path.join(out_dir, f"{name}.wav")
    sf.write(wav, mix if mix.ndim == 2 else np.stack([mix, mix], 1), A.SR)
    write_srt(scenes, os.path.join(out_dir, f"{name}.srt"))
    nframes = int(math.ceil(total * fps))
    workers = workers or max(1, os.cpu_count() or 1)
    # split at roughly equal frame counts
    bounds = [round(nframes * k / workers) for k in range(workers + 1)]
    segdir = os.path.join(out_dir, "_segments")
    os.makedirs(segdir, exist_ok=True)
    jobs = [(bounds[k], bounds[k + 1], os.path.join(segdir, f"{name}_{k:02d}.mp4"), fps) for k in range(workers)]
    ctx_data = dict(story=story, scenes=scenes, W=W, H=H, cache=CACHE)
    mpctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    done = {}
    try:
        with ProcessPoolExecutor(workers, mp_context=mpctx, initializer=_init_worker, initargs=(ctx_data,)) as ex:
            futs = {ex.submit(_worker, j): k for k, j in enumerate(jobs)}
            for fu, k in futs.items():
                done[k] = fu.result()
    except BrokenProcessPool:
        print("  a render worker died (out of memory?); finishing the remaining segments one at a time", flush=True)
    for k, j in enumerate(jobs):
        if k not in done:
            with ProcessPoolExecutor(1, mp_context=mpctx, initializer=_init_worker, initargs=(ctx_data,)) as ex:
                done[k] = ex.submit(_worker, j).result()
    segs = [done[k] for k in range(len(jobs))]
    lst = os.path.join(segdir, f"{name}_list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)
    silent = os.path.join(segdir, f"{name}_silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent], check=True)
    out = os.path.join(out_dir, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    fm = FrameMaker(story, scenes, W // 2, H // 2, CACHE)
    contact_sheet(fm, scenes, os.path.join(out_dir, f"{name}.contact.jpg"))
    print(f"[{name}] done: {out} ({total:.1f}s video) in {time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a storyboard JSON to MP4")
    r.add_argument("story")
    r.add_argument("--out", default="out")
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fps", type=int, default=24)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--preview", action="store_true", help="only write a contact sheet")
    r.add_argument("--stills", type=str, default=None, help="comma list of scene fractions to dump, e.g. 0.2,0.8")
    r.add_argument("--only", type=str, default=None, help="comma list of scene indices for --stills")
    m = sub.add_parser("make", help="story text -> storyboard (Claude) -> MP4")
    m.add_argument("story_txt")
    m.add_argument("--title", required=True)
    m.add_argument("--subtitle")
    m.add_argument("--lang", default="en", choices=["en", "zh"])
    m.add_argument("--scenes", default="10-14")
    m.add_argument("--backend", choices=["claude-cli", "api"])
    m.add_argument("--model")
    m.add_argument("--critique", type=int, default=1)
    m.add_argument("--out", default="out")
    m.add_argument("--height", type=int, default=1080)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        from .storyboard import generate
        slug = "".join(ch if ch.isalnum() else "_" for ch in a.title.lower()).strip("_") or "story"
        sb_path = os.path.join(a.out, f"{slug}.json")
        text = open(a.story_txt, encoding="utf-8").read()
        generate(text, a.title, a.lang, a.subtitle, a.scenes, a.backend, a.model, sb_path, critique=a.critique)
        render(sb_path, a.out, a.height)
        return
    if a.cmd == "render":
        stills = [float(x) for x in a.stills.split(",")] if a.stills else None
        only = [int(x) for x in a.only.split(",")] if a.only else None
        render(a.story, a.out, a.height, a.fps, a.workers, a.preview, stills, only)


if __name__ == "__main__":
    main()
