"""Rigging: cut pieces of generated art -> storyvid actors that drop into an existing storyboard.

    python -m storyvid.rig hybrid/cast/jingwei.json --assets generated --out out/jingwei_hybrid

A cast file maps each actor a storyboard uses to generated art:

    "jingwei": {"kind": "figure", "sheet": "jingwei_step1/cut/bird_parts", "root": 2,
                "parts": {"wing_f": {"piece": 0, "attach": [0.36, 0.13]}, ...},
                "fit": {"width": 350}, "anchor": [0.5, 1.0]}
    "boat":    {"kind": "prop", "sheet": "jingwei_film/cut/props_sea", "piece": 0,
                "fit": {"width": 366}, "anchor": [0.5, 1.0]}
    "sea":     {"kind": "layers", "layers": {"w1": {"image": "jingwei_film/sea_back.png", "top": 0}, ...},
                "x": [-300, 2220], "bottom": 760}

Figures: every limb's joint is found on its own (the straight cut edge, which the image model leaves where a piece
was cut from the body); the limb is then moved so that joint sits on its attach point, given as a fraction of the root
piece's box (the planner's call). Optional per part: "flip", "rot" (degrees, before placing), "z" (draw order),
"hidden", "offset" (fine adjustment in sheet pixels). The assembled figure is scaled to "fit" (width or height in
storyboard units) and placed so "anchor" (a fraction of its box) is the actor origin, matching the procedural actor
it replaces. Rigs are copied from that actor; entries for parts the figure does not have are dropped (and listed),
or redirected with "alias": {"crown": "tree"}.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------- pieces and joints

def load_sheet(assets_root: str, sheet: str) -> dict:
    d = os.path.join(assets_root, sheet)
    with open(os.path.join(d, "report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    return {"dir": d, "pieces": {p["index"]: p for p in rep["pieces"]}}


def read_rgba(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return im


def joint_edge(rgba: np.ndarray, near=None):
    """The cut where a limb was taken off the body: (p0, p1, pivot) in piece pixels, or None.

    Pieces painted for a parts sheet are cut straight across where they join the body. Among the straight stretches of
    the outline, the joint is the one that runs across the piece's long axis and sits at one end of it (a wing's
    leading edge is just as straight, but it runs along the axis). Its midpoint, pulled slightly inside, is the pivot.
    `near` ([u, v] as a fraction of the piece's box) says which end the joint is at when both ends are cut straight.
    """
    mask = (rgba[..., 3] > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    n = len(c)
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float64)
    cen = P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((P - cen).T))
    u, v = evecs[:, 1], evecs[:, 0]  # long axis, short axis
    t = (P - cen) @ u
    tmin, tmax = t.min(), t.max()
    width = np.ptp((P - cen) @ v)
    best, best_score = None, 0.0
    step = max(1, n // 500)
    for span_frac in (0.04, 0.07, 0.1, 0.14):
        span = max(8, int(n * span_frac))
        for i in range(0, n, step):
            j = (i + span) % n
            a, b = c[i], c[j]
            d = b - a
            L = float(np.hypot(*d))
            if L < 0.25 * width:
                continue
            seg = c[np.arange(i, i + span + 1) % n] - a
            dev = np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / L
            if dev.max() > max(2.5, 0.03 * L):
                continue
            across = 1.0 - abs(float(d @ u)) / L
            tm = float(((a + b) / 2 - cen) @ u)
            endness = tm / tmax if tm > 0 else tm / tmin
            score = across ** 2 * max(0.0, endness) ** 3 * min(1.0, L / (0.6 * width))
            if near is not None:
                hint = np.array([near[0] * mask.shape[1], near[1] * mask.shape[0]])
                score *= math.exp(-float(np.hypot(*((a + b) / 2 - hint))) / (0.35 * max(mask.shape)))
            if score > best_score:
                best, best_score = (a, b), score
    if best is None or best_score < 0.15:
        return None
    a, b = best
    mid = (a + b) / 2
    L = float(np.hypot(*(b - a)))
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / max(L, 1e-9)
    if (cen - mid) @ nrm < 0:
        nrm = -nrm
    return a, b, mid + nrm * 0.08 * L


def transform_piece(rgba: np.ndarray, flip: bool = False, rot: float = 0.0):
    """Mirror and/or rotate a piece (degrees, counter-clockwise on screen). -> (image, 2x3 matrix old px -> new px)."""
    h, w = rgba.shape[:2]
    M = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip:
        M = np.array([[-1.0, 0, w - 1], [0, 1, 0], [0, 0, 1]]) @ M
    if rot:
        R = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), rot, 1.0)
        cos, sin = abs(R[0, 0]), abs(R[0, 1])
        nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
        R[0, 2] += nw / 2 - (w - 1) / 2
        R[1, 2] += nh / 2 - (h - 1) / 2
        M = np.vstack([R, [0, 0, 1]]) @ M
        size = (nw, nh)
    else:
        size = (w, h)
    out = cv2.warpAffine(rgba, M[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, M


def _apply(M, p):
    return (M @ np.array([p[0], p[1], 1.0]))[:2]


# --------------------------------------------------------------------------- actors

def _fit_and_anchor(boxes: list, fit: dict, anchor) -> tuple[float, np.ndarray]:
    """Scale (sheet px -> units) and translation so the union box matches fit/anchor."""
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    if "width" in fit:
        k = fit["width"] / (x1 - x0)
    else:
        k = fit["height"] / (y1 - y0)
    ax, ay = anchor
    ox = x0 + ax * (x1 - x0)
    oy = y0 + ay * (y1 - y0)
    return k, np.array([ox, oy])


def build_figure(name: str, spec: dict, assets_root: str, out_dir: str, log=print) -> dict:
    sheet = load_sheet(assets_root, spec["sheet"])
    root = sheet["pieces"][spec["root"]]
    root_img = read_rgba(os.path.join(sheet["dir"], root["file"]))
    rh, rw = root_img.shape[:2]
    ox, oy = root["origin"]
    placed = [{"id": spec.get("root_id", "body"), "img": root_img, "pos": np.array([ox, oy], float), "pivot": None,
               "z": 0, "hidden": False, "parent": None}]
    for pid, ps in spec["parts"].items():
        if "piece_of" in ps:  # a piece from another sheet (e.g. a prop held in the beak)
            other = load_sheet(assets_root, ps["piece_of"])
            pc = other["pieces"][ps["piece"]]
            src_dir = other["dir"]
        else:
            pc = sheet["pieces"][ps["piece"]]
            src_dir = sheet["dir"]
        img = read_rgba(os.path.join(src_dir, pc["file"]))
        if "pivot_at" in ps:  # the planner's pivot, as a fraction of the piece box (e.g. a rounded shoulder)
            piv = np.array([ps["pivot_at"][0] * img.shape[1], ps["pivot_at"][1] * img.shape[0]])
        else:
            j = joint_edge(img, ps.get("joint_near")) if ps.get("joint", True) else None
            piv = j[2] if j is not None else np.array([img.shape[1] / 2, img.shape[0] / 2])
            if j is None and ps.get("joint", True):
                log(f"  [{name}] {pid}: no straight joint edge found; pivot at the piece centre")
        if "scale" in ps:
            img = cv2.resize(img, None, fx=ps["scale"], fy=ps["scale"], interpolation=cv2.INTER_AREA)
            piv = piv * ps["scale"]
        img2, M = transform_piece(img, ps.get("flip", False), ps.get("rot", 0.0))
        piv2 = _apply(M, piv)
        ax, ay = ps["attach"]
        target = np.array([ox + ax * rw, oy + ay * rh]) + np.array(ps.get("offset", [0, 0]), float)
        placed.append({"id": pid, "img": img2, "pos": target - piv2, "pivot": target, "z": ps.get("z", 1),
                       "hidden": ps.get("hidden", False), "parent": ps.get("parent", placed[0]["id"])})
    boxes = [(p["pos"][0], p["pos"][1], p["img"].shape[1], p["img"].shape[0]) for p in placed if not p["hidden"]]
    k, org = _fit_and_anchor(boxes, spec["fit"], spec.get("anchor", [0.5, 1.0]))
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    parts = []
    for p in sorted(placed, key=lambda q: q["z"]):
        fname = f"parts/{name}_{p['id']}.png"
        cv2.imwrite(os.path.join(out_dir, fname), p["img"])
        h, w = p["img"].shape[:2]
        x, y = (p["pos"] - org) * k
        part = {"id": p["id"], "shape": "image", "src": fname, "xywh": [round(x, 2), round(y, 2), round(w * k, 2), round(h * k, 2)]}
        if p["pivot"] is not None:
            part["pivot"] = [round(v, 2) for v in ((p["pivot"] - org) * k)]
        if p["parent"] and p["id"] != placed[0]["id"]:
            part["parent"] = p["parent"]
        if p["hidden"]:
            part["hidden"] = True
        parts.append(part)
    return {"parts": parts}


def build_prop(name: str, spec: dict, assets_root: str, out_dir: str) -> dict:
    if "region" in spec:  # a box of a sheet, keyed as a whole (an object made of several loose bits, e.g. a sun's rays)
        from .cutout import key_mask
        src = cv2.imread(os.path.join(assets_root, spec["image"]), cv2.IMREAD_COLOR)
        H, W = src.shape[:2]
        x0, y0, x1, y1 = spec["region"]
        crop = src[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        alpha, solid, _, _ = key_mask(crop)
        ys, xs = np.nonzero(solid)
        crop, alpha = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = np.dstack([crop, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
    elif "image" in spec:  # a whole image (e.g. a strip or a backdrop) rather than a cut piece
        img = read_rgba(os.path.join(assets_root, spec["image"]))
    else:
        sheet = load_sheet(assets_root, spec["sheet"])
        pc = sheet["pieces"][spec["piece"]]
        img = read_rgba(os.path.join(sheet["dir"], pc["file"]))
    img, _ = transform_piece(img, spec.get("flip", False), spec.get("rot", 0.0))
    h, w = img.shape[:2]
    k, org = _fit_and_anchor([(0, 0, w, h)], spec["fit"], spec.get("anchor", [0.5, 0.5]))
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    fname = f"parts/{name}.png"
    cv2.imwrite(os.path.join(out_dir, fname), img)
    pid = spec.get("id", name)
    x, y = (-org) * k
    part = {"id": pid, "shape": "image", "src": fname, "xywh": [round(x, 2), round(y, 2), round(w * k, 2), round(h * k, 2)]}
    if "shadow" in spec:
        part["shadow"] = spec["shadow"]
    if "pivot" in spec:
        part["pivot"] = spec["pivot"]
    return {"parts": [part]}


def build_layers(name: str, spec: dict, assets_root: str, out_dir: str) -> dict:
    """Full-width strips (sea, ground): each layer's cut piece is stretched across x and hung from its `top`."""
    x0, x1 = spec.get("x", [-300, 2220])
    parts = []
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    for lid, ls in spec["layers"].items():
        if "sheet" in ls:
            sheet = load_sheet(assets_root, ls["sheet"])
            pc = sheet["pieces"][ls.get("piece", 0)]
            img = read_rgba(os.path.join(sheet["dir"], pc["file"]))
        else:
            img = read_rgba(os.path.join(assets_root, ls["image"]))
        a = img[..., 3] > 127
        ys = np.where(a.any(1))[0]
        img = img[ys[0]:ys[-1] + 1]
        h, w = img.shape[:2]
        k = (x1 - x0) / w
        top = ls.get("top", 0)
        hh = max(h * k, spec.get("bottom", 760) - top) if spec.get("fill_down", True) else h * k
        fname = f"parts/{name}_{lid}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        parts.append({"id": lid, "shape": "image", "src": fname, "xywh": [x0, top - ls.get("crest", 0.0) * h * k, x1 - x0, round(hh, 2)],
                      "pivot": [round((x0 + x1) / 2, 2), top], "shadow": ls.get("shadow", 0.35)})
    return {"parts": parts}


def adopt_rigs(name: str, actor: dict, original: dict | None, alias: dict, log=print) -> dict:
    """Copy the replaced actor's rigs, keeping only entries whose parts exist (after aliasing)."""
    if not original or not original.get("rigs"):
        return actor
    have = {p["id"] for p in actor["parts"]}
    rigs, dropped = {}, set()
    for rname, entries in original["rigs"].items():
        out = []
        for e in entries:
            targets = e["part"] if isinstance(e["part"], list) else [e["part"]]
            mapped = []
            for t in targets:
                t2 = alias.get(t, t)
                if t2 in have:
                    mapped.append(t2)
                else:
                    dropped.add(t)
            if mapped:
                e2 = dict(e, part=mapped if len(mapped) > 1 else mapped[0])
                out.append(e2)
        rigs[rname] = out
    actor["rigs"] = rigs
    if dropped:
        log(f"  [{name}] rig entries dropped for parts it does not have: {', '.join(sorted(dropped))}")
    return actor


# --------------------------------------------------------------------------- storyboard swap

def hybridize(story: dict, cast: dict, assets_root: str, out_dir: str, log=print) -> dict:
    """-> a copy of `story` whose actors are generated art and whose scene backgrounds are generated backdrops."""
    from . import pipeline as P
    originals = {}
    built = P.build_actors(story)
    for n, a in built.items():
        originals[n] = {"parts": a.parts, "rigs": a.rigs}
    new = copy.deepcopy(story)
    new.setdefault("actors", {})
    have_parts = {}
    for name, spec in cast.get("actors", {}).items():
        kind = spec.get("kind", "prop")
        if kind == "figure":
            actor = build_figure(name, spec, assets_root, out_dir, log)
        elif kind == "layers":
            actor = build_layers(name, spec, assets_root, out_dir)
        else:
            actor = build_prop(name, spec, assets_root, out_dir)
        actor = adopt_rigs(name, actor, originals.get(name), spec.get("alias", {}), log)
        if spec.get("rigs"):  # tuned motion for painted pieces replaces the procedural values rig by rig
            actor.setdefault("rigs", {}).update(spec["rigs"])
        # who it depicts carries over to the generated actor; library scenery (sea, sun, ...) is never a character
        actor["character"] = story.get("actors", {}).get(name, {}).get("character", False)
        new["actors"][name] = actor
        have_parts[name] = {p["id"] for p in actor["parts"]}

    def fix_item(it):
        n = it.get("actor")
        if n not in have_parts:
            return
        alias = cast["actors"][n].get("alias", {})
        have = have_parts[n]
        for key in ("hide", "reveal"):
            if key in it:
                it[key] = [alias.get(p, p) for p in it[key] if alias.get(p, p) in have]
        if "part_anims" in it:
            kept = []
            for pa in it["part_anims"]:
                ts = pa["part"] if isinstance(pa["part"], list) else [pa["part"]]
                ts = [alias.get(t, t) for t in ts if alias.get(t, t) in have]
                if ts:
                    kept.append(dict(pa, part=ts if len(ts) > 1 else ts[0]))
            it["part_anims"] = kept
        r = it.get("rig")
        if r:
            rigs = new["actors"][n].get("rigs", {})
            names = r if isinstance(r, list) else [r]
            names = [x for x in names if (x["name"] if isinstance(x, dict) else x) in rigs]
            if names:
                it["rig"] = names if isinstance(r, list) else names[0]
            else:
                it.pop("rig")

    backdrops = cast.get("backdrops", {})
    scenes = list(new["scenes"]) + [s for s in (new.get("title_scene"), new.get("end_scene")) if s]
    for sc in scenes:
        for it in sc.get("items", []):
            fix_item(it)
        bg = sc.get("background")
        if bg in backdrops:
            sc["background"] = None
            cam = sc.get("camera") or {}
            z = cam.get("zoom", [1.0, 1.035])
            zmax = max(z) if not np.isscalar(z) else z
            sc.setdefault("items", []).insert(0, {"part": {"shape": "image", "src": backdrops[bg], "xywh": [-160, -120, 2240, 1320],
                                                           "shadow": 0}, "z": -1000, "detail": 1.0 / zmax})
    for k, v in cast.get("story", {}).items():
        new[k] = v
    return new


def preview(story: dict, names: list[str], path: str, tile=(480, 300)):
    """Joint check: each actor at four moments of its first rig, one row per actor (story loaded with load_story)."""
    from . import pipeline as P
    from PIL import Image, ImageDraw
    st = copy.deepcopy(story)
    st["scenes"] = []
    for n in names:
        a = st["actors"][n]
        rig_name = next(iter(a.get("rigs", {})), None)
        item = {"actor": n, "at": [960, 760], "scale": 1.0}
        if rig_name:
            item["rig"] = rig_name
        st["scenes"].append({"narration": "", "items": [item], "camera": {"zoom": 1.0}})
    st["title_card"] = st["end_card"] = False
    scenes = P.build_scenes(st)
    for sc in scenes:
        sc["_dur"], sc["_captions"], sc["_start"] = 4.0, [], 0.0
    fm = P.FrameMaker(st, scenes, 960, 540, P.CACHE)
    rows = []
    for i, n in enumerate(names):
        frames = []
        for u in (0.05, 0.3, 0.55, 0.8):
            f = (np.clip(fm.scene_frame(i, 4.0 * u), 0, 1) * 255).astype(np.uint8)
            frames.append(Image.fromarray(f).resize(tile))
        row = Image.new("RGB", (tile[0] * 4, tile[1] + 22), "white")
        for k, fr in enumerate(frames):
            row.paste(fr, (k * tile[0], 22))
        ImageDraw.Draw(row).text((6, 4), f"{n}  rig: {next(iter(st['actors'][n].get('rigs', {})), '-')}", fill=(30, 30, 30))
        rows.append(row)
    sheet = Image.new("RGB", (tile[0] * 4, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(path, quality=88)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cast")
    ap.add_argument("--assets", default="generated", help="folder holding the generated runs (e.g. a checkout of generated-assets)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.cast, encoding="utf-8") as f:
        cast = json.load(f)
    story_path = os.path.join(os.path.dirname(os.path.abspath(a.cast)), cast["storyboard"])
    with open(story_path, encoding="utf-8") as f:
        story = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    assets = os.path.abspath(a.assets)
    for k, v in cast.get("backdrops", {}).items():  # copy backdrops next to the storyboard
        src = os.path.join(assets, v)
        dst = os.path.join("parts", "bg_" + k + ".png")
        os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
        cv2.imwrite(os.path.join(a.out, dst), cv2.imread(src, cv2.IMREAD_UNCHANGED))
        cast["backdrops"][k] = dst
    new = hybridize(story, cast, assets, a.out)
    if cast.get("soundtrack"):  # generated narration and music (storyvid.soundtrack): copy next to the storyboard
        import shutil
        shutil.copytree(os.path.join(assets, cast["soundtrack"]), os.path.join(a.out, "soundtrack"), dirs_exist_ok=True)
        new["soundtrack"] = "soundtrack"
    out = os.path.join(a.out, os.path.splitext(os.path.basename(story_path))[0] + "_hybrid.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    print(f"[rig] wrote {out} ({len(cast.get('actors', {}))} actors from generated art)")


if __name__ == "__main__":
    main()
