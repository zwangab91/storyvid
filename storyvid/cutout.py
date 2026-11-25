"""Cut paper pieces off a plain board: board keying, connected pieces, RGBA crops.

    python -m storyvid.cutout generated/jingwei_step1/*.png      # -> generated/jingwei_step1/cut/<image>/

The board colour and its noise are measured on the image border. A pixel belongs to a piece when its colour is far
enough from the board (CIE Lab distance with a soft ramp). Holes enclosed by a piece are filled, since paper pieces
are opaque even where their paint is close to the board colour. Each connected piece above a minimum area is
written as an RGBA crop, with an overlay and a JSON report that say whether the image is cleanly cuttable:
plain board, no piece cut off by the image edge (full-width scenery strips may touch the left and right
edges), no near-board-coloured paint just inside a piece's outline.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def _lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def board_stats(lab: np.ndarray, band: float = 0.03):
    """Median board colour and robust spread, measured on the outer band of the image."""
    h, w = lab.shape[:2]
    b = max(2, int(round(min(h, w) * band)))
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[b:-b, :b].reshape(-1, 3), lab[b:-b, -b:].reshape(-1, 3)])
    med = np.median(ring, axis=0)
    dist = np.linalg.norm(ring - med, axis=1)
    mad = float(np.median(dist))
    clean = float(np.mean(dist < max(6.0, 4 * mad)))  # share of the border that looks like board
    return med, mad, clean


def key_mask(img_bgr: np.ndarray, min_area_frac: float = 0.0004):
    """-> (alpha float32 [0,1], board info dict)."""
    lab = _lab(img_bgr)
    med, mad, clean = board_stats(lab)
    d = np.linalg.norm(lab - med, axis=2)
    t0 = max(5.0, 3.5 * mad)
    t1 = t0 + 7.0
    soft = np.clip((d - t0) / (t1 - t0), 0, 1).astype(np.float32)
    hard = (soft > 0.5).astype(np.uint8)
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill enclosed holes: everything not reachable from the border through board pixels is piece
    h, w = hard.shape
    flood = hard.copy() * 255
    pad = np.zeros((h + 2, w + 2), np.uint8)
    for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if flood[y, x] == 0:
            cv2.floodFill(flood, pad, (x, y), 128)
    solid = (flood != 128).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area_frac * h * w
    solid = keep[lbl].astype(np.uint8)
    # soft edge: inside the solid region use full opacity, at the boundary use the colour ramp
    inner = cv2.erode(solid, np.ones((3, 3), np.uint8))
    alpha = np.where(inner > 0, 1.0, np.where(solid > 0, np.maximum(soft, 0.5), 0.0)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7) * (cv2.dilate(solid, np.ones((3, 3), np.uint8)) > 0)
    info = {"board_lab": [round(float(v), 2) for v in med], "board_noise": round(mad, 2),
            "border_is_board": round(clean, 3), "threshold": round(t0, 2)}
    return alpha, solid, d, info


def pieces(solid: np.ndarray, d: np.ndarray, info: dict) -> list[dict]:
    h, w = solid.shape
    n, lbl, st, cen = cv2.connectedComponentsWithStats(solid, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        comp = (lbl == i).astype(np.uint8)
        # paint just inside the outline (3-7 px in) that is close to the board colour: the cut there is a guess
        depth = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        band = (depth >= 3) & (depth <= 7)
        weak = float(np.mean(d[band] < info["threshold"] + 7.0)) if band.any() else 0.0
        out.append({"index": len(out), "bbox": [x, y, bw, bh], "area_frac": round(area / (h * w), 4),
                    "centroid": [round(float(cen[i][0]), 1), round(float(cen[i][1]), 1)],
                    "touches_edge": bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1),
                    "edges": [n for n, hit in (("top", y <= 1), ("bottom", y + bh >= h - 1), ("left", x <= 1),
                                                ("right", x + bw >= w - 1)) if hit],
                    "weak_outline": round(weak, 3), "_label": i})
    out.sort(key=lambda p: (p["bbox"][1] // max(1, h // 6), p["bbox"][0]))  # reading order, roughly
    for k, p in enumerate(out):
        p["index"] = k
    return out, lbl


def cut(path: str, out_dir: str, pad: int = 6) -> dict:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    alpha, solid, d, info = key_mask(img)
    ps, lbl = pieces(solid, d, info)
    os.makedirs(out_dir, exist_ok=True)
    h, w = solid.shape
    overlay = img.copy()
    for p in ps:
        x, y, bw, bh = p["bbox"]
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
        own = cv2.dilate((lbl == p["_label"]).astype(np.uint8), np.ones((5, 5), np.uint8))
        a = (alpha * own)[y0:y1, x0:x1]
        rgba = np.dstack([img[y0:y1, x0:x1], np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)])
        p["file"] = f"piece_{p['index']:02d}.png"
        p["origin"] = [x0, y0]
        cv2.imwrite(os.path.join(out_dir, p["file"]), rgba)
        cnts, _ = cv2.findContours((lbl == p["_label"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = (0, 0, 255) if p["touches_edge"] or p["weak_outline"] > 0.15 else (0, 160, 0)
        cv2.drawContours(overlay, cnts, -1, col, 3)
        cv2.putText(overlay, str(p["index"]), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        del p["_label"]
    cv2.imwrite(os.path.join(out_dir, "overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    def bad_edge(p):
        full_width = p["bbox"][2] >= 0.9 * w
        return bool(set(p["edges"]) - ({"left", "right"} if full_width else set()))
    ok = (info["border_is_board"] > 0.97 and not any(bad_edge(p) for p in ps)
          and all(p["weak_outline"] <= 0.15 for p in ps) and len(ps) > 0)
    report = {"image": os.path.basename(path), "size": [w, h], "board": info, "n_pieces": len(ps),
              "cleanly_cuttable": bool(ok), "pieces": ps}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="folder to write cut/<image name>/ into (default: next to the image)")
    a = ap.parse_args(argv)
    for p in a.images:
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(a.out or os.path.dirname(p) or ".", "cut", stem)
        r = cut(p, out)
        flag = "clean" if r["cleanly_cuttable"] else "needs a look"
        print(f"{stem}: {r['n_pieces']} pieces, board {r['board']['border_is_board']:.0%} plain, {flag} -> {out}")


if __name__ == "__main__":
    main()
