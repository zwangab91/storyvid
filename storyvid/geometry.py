"""Shape primitives -> polygons (design units, y down), plus cut-paper edge roughening."""
from __future__ import annotations

import math

import numpy as np


def _arr(p):
    return np.asarray(p, np.float64)


# --------------------------------------------------------------------------- splines

def _cr_segment(p0, p1, p2, p3, n, alpha=0.5):
    def tj(ti, a, b):
        return ti + max(float(np.linalg.norm(b - a)), 1e-4) ** alpha
    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False)[:, None]
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def spline(points, closed=True, corners=(), density=3.0):
    """Centripetal Catmull-Rom through `points`. Indices in `corners` stay sharp."""
    P = _arr(points)
    n = len(P)
    if n < 3:
        return P
    corners = set(i % n for i in corners)
    out = []
    segs = n if closed else n - 1
    for i in range(segs):
        i1, i2 = i, (i + 1) % n
        p1, p2 = P[i1], P[i2]
        if closed:
            p0 = P[(i - 1) % n] if i1 not in corners else 2 * p1 - p2
            p3 = P[(i + 2) % n] if i2 not in corners else 2 * p2 - p1
        else:
            p0 = P[i - 1] if (i > 0 and i1 not in corners) else 2 * p1 - p2
            p3 = P[i + 2] if (i + 2 < n and i2 not in corners) else 2 * p2 - p1
        L = float(np.linalg.norm(p2 - p1))
        k = max(3, int(L / density))
        out.append(_cr_segment(p0, p1, p2, p3, k))
    if not closed:
        out.append(P[-1:])
    return np.concatenate(out)


# --------------------------------------------------------------------------- primitives

def ellipse(c, r, rot=0.0, arc=None):
    cx, cy = c
    rx, ry = (r, r) if np.isscalar(r) else r
    n = max(24, int(2 * math.pi * max(rx, ry) / 3))
    if arc:  # pie slice [a0, a1] degrees
        a = np.radians(np.linspace(arc[0], arc[1], n))
        pts = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
        pts = np.vstack([[0, 0], pts])
    else:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    return _rot(pts, rot) + [cx, cy]


def rect(c=None, size=None, xywh=None, radius=0.0, rot=0.0):
    if xywh is not None:
        x, y, w, h = xywh
        c = (x + w / 2, y + h / 2)
        size = (w, h)
    w, h = size
    r = min(radius, w / 2, h / 2)
    if r <= 0:
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    else:
        pts = []
        for cx, cy, a0 in [(w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                           (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)]:
            a = np.radians(np.linspace(a0, a0 + 90, 8))
            pts += list(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
        pts = np.array(pts)
    return _rot(pts, rot) + c


def star(c, r, n=5, rot=0.0, smooth=False):
    rin, rout = r
    pts = []
    for i in range(2 * n):
        a = math.radians(rot - 90 + i * 180 / n)
        rr = rout if i % 2 == 0 else rin
        pts.append([c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)])
    return spline(pts) if smooth else np.array(pts)


def blob(c, r, jag=0.18, n=9, seed=0, rot=0.0):
    rng = np.random.default_rng(seed)
    rx, ry = (r, r) if np.isscalar(r) else r
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(-0.25, 0.25, n) * (2 * math.pi / n)
    k = 1 + rng.uniform(-jag, jag, n)
    pts = np.stack([rx * k * np.cos(a), ry * k * np.sin(a)], 1)
    return spline(_rot(pts, rot) + c)


def cloud(c, r, puffs=6, seed=0):
    """Puffy cloud: outline of a union of circles with a flat-ish bottom."""
    import cv2
    rng = np.random.default_rng(seed)
    rx, ry = (r, r * 0.55) if np.isscalar(r) else r
    S = 4.0  # raster units per design unit for contour extraction
    W, H = int(rx * 2.6 * S) + 8, int(ry * 2.8 * S) + 8
    m = np.zeros((H, W), np.uint8)
    ox, oy = W / 2, H * 0.58
    xs = np.linspace(-rx * 0.62, rx * 0.62, puffs)
    for i, x in enumerate(xs):
        k = 1 - (abs(x) / rx) ** 2 * 0.9
        pr = ry * (0.45 + 0.5 * k) * rng.uniform(0.85, 1.1)
        y = -pr * 0.35 * rng.uniform(0.6, 1.2)
        cv2.circle(m, (int(ox + x * S), int(oy + y * S)), int(pr * S), 255, -1)
    cv2.ellipse(m, (int(ox), int(oy + ry * 0.05 * S)), (int(rx * 0.8 * S), int(ry * 0.42 * S)), 0, 0, 360, 255, -1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    P = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    P = (P - [ox, oy]) / S + c
    return densify_closed(P, 4.0)


def stroke(points, width, smooth=True, cap="round"):
    """Painted line of given width (or [w_start, w_end] taper) -> closed polygon."""
    P = spline(points, closed=False) if (smooth and len(points) > 2) else _densify_open(_arr(points), 3)
    w0, w1 = (width, width) if np.isscalar(width) else width
    d = np.diff(P, axis=0)
    seg = np.linalg.norm(d, axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    P = P[keep]
    if len(P) < 2:
        P = np.vstack([P, P + [0.01, 0]])
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = s / max(s[-1], 1e-6)
    w = (w0 + (w1 - w0) * t)[:, None] / 2
    left, right = P + nrm * w, P - nrm * w
    out = [left]
    if cap == "round" and w1 > 0.5:
        a0 = math.atan2(nrm[-1, 1], nrm[-1, 0])
        a = np.linspace(a0, a0 - math.pi, 10)[1:-1]
        out.append(P[-1] + np.stack([np.cos(a), np.sin(a)], 1) * w[-1])
    out.append(right[::-1])
    if cap == "round" and w0 > 0.5:
        a0 = math.atan2(-nrm[0, 1], -nrm[0, 0])
        a = np.linspace(a0, a0 - math.pi, 10)[1:-1]
        out.append(P[0] + np.stack([np.cos(a), np.sin(a)], 1) * w[0])
    return np.concatenate(out)


def wave_band(x0, x1, y, amp, wavelength, bottom, phase=0.0, seed=0):
    """A band whose top edge is a wobbly wave (sea, hills). Returns polygon."""
    rng = np.random.default_rng(seed)
    xs = np.arange(x0, x1 + wavelength / 8, wavelength / 8)
    ph2 = rng.uniform(0, 6.28)
    ys = y - amp * np.sin(2 * math.pi * (xs - x0) / wavelength + phase) \
        - 0.25 * amp * np.sin(2 * math.pi * (xs - x0) / (wavelength * 0.43) + ph2)
    top = np.stack([xs, ys], 1)
    top = spline(top, closed=False)
    return np.vstack([top, [[x1, bottom], [x0, bottom]]])


def _rot(pts, deg):
    if not deg:
        return pts
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return pts @ R.T


def _densify_open(P, step):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(b - a))
        k = max(1, int(L / step))
        for i in range(1, k + 1):
            out.append(a + (b - a) * i / k)
    return np.array(out)


def densify_closed(P, step):
    Q = np.vstack([P, P[:1]])
    return _densify_open(Q, step)[:-1]


# --------------------------------------------------------------------------- shape dispatch

def shape_polygon(p: dict) -> np.ndarray:
    s = p.get("shape", "poly")
    rot = float(p.get("rot", 0.0))
    if s in ("ellipse", "circle"):
        r = p.get("r", 50)
        return ellipse(p.get("c", (0, 0)), r, rot, p.get("arc"))
    if s == "rect":
        if "xywh" in p:
            return rect(xywh=p["xywh"], radius=p.get("radius", 0), rot=rot)
        return rect(p.get("c", (0, 0)), p.get("size", (100, 100)), radius=p.get("radius", 0), rot=rot)
    if s == "poly":
        pts = p["points"]
        if p.get("smooth", False):
            return spline(pts, closed=True, corners=p.get("corners", ()))
        return _arr(pts)
    if s == "blob":
        return blob(p.get("c", (0, 0)), p.get("r", 50), p.get("jag", 0.18), p.get("n", 9), p.get("shape_seed", 0), rot)
    if s == "cloud":
        return cloud(p.get("c", (0, 0)), p.get("r", 120), p.get("puffs", 6), p.get("shape_seed", 0))
    if s == "star":
        return star(p.get("c", (0, 0)), p.get("r", (30, 70)), p.get("n", 5), rot, p.get("smooth", False))
    if s == "stroke":
        return stroke(p["points"], p.get("width", 12), p.get("smooth", True), p.get("cap", "round"))
    if s == "wave":
        return wave_band(p.get("x0", -200), p.get("x1", 2120), p.get("y", 700), p.get("amp", 30),
                         p.get("wavelength", 300), p.get("bottom", 1280), p.get("phase", 0.0), p.get("shape_seed", 0))
    raise ValueError(f"unknown shape {s!r}")


def roughen(P, amp, seed=0, freq=1.0):
    """Displace a closed polygon along its normals -> scissor-cut / torn paper edge."""
    if amp <= 0 or len(P) < 3:
        return P
    Q = densify_closed(P, 2.5)
    n = len(Q)
    rng = np.random.default_rng(seed)
    d = np.roll(Q, -1, 0) - np.roll(Q, 1, 0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)

    def circ_noise(sig):
        sig = min(sig, n / 6.0)  # small shapes: keep a few wobbles around the outline
        x = rng.standard_normal(n)
        k = np.fft.rfftfreq(n)
        X = np.fft.rfft(x) * np.exp(-0.5 * (k * sig * 2 * math.pi) ** 2)  # gaussian low-pass
        X[0] = 0.0  # no net inflation
        y = np.fft.irfft(X, n)
        return y / max(y.std(), 1e-3)
    disp = 0.75 * circ_noise(14 / freq) + 0.25 * circ_noise(2.0 / freq)
    return Q + nrm * (amp * disp)[:, None]
