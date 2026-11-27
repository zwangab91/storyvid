"""Video clips from Gemini Omni for moments a rig can't act: a manifest of start frames + motion directions -> mp4s.

A clip starts from one of our own rendered frames (no captions on it), so the characters and paper look are the
film's, and asks for exactly the length of the moment (3 to 15 s), so nothing has to loop:

    {"kind": "video", "model": "gemini-omni-1.2-flash", "resolution": "720p", "aspect_ratio": "16:9",
     "style": "...style lock appended to every prompt...",
     "assets": [{"id": "drink", "image": "frames/kuafu_s8.jpg", "duration": 15,
                 "prompt": "Motion direction: the giant kneels at the riverbank and drinks ..."}]}

`image` is a path relative to the manifest (or "@run/file" in the assets folder, like image refs); `last_image`
(optional, 3-10 s only) asks Omni to end on that frame, so the clip can cut back into the puppet animation. Run it
like any manifest ("Generate assets" workflow): results are cached by request, capped by --max-images (one clip each),
cost-logged in assets.json. Omni returns native audio; storyvid ignores it and keeps its own narration and music.

Pricing (Gemini API, standard tier, Nov 2026): video output $12 per million tokens at about 5,792 tokens per second
of 720p, i.e. about $0.07 a second; input $1.50 per million tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time

INTERACTIONS_API = "https://generativelanguage.googleapis.com/v1beta/interactions"
VIDEO_PRICE = {"gemini-omni-1.2-flash": {"in": 1.50, "video_out": 12.00, "text_out": 9.00, "tokens_per_s": 5792}}
DEFAULT_STYLE = ("Style lock: preserve every scissor-cut paper edge, painted brushstroke and paper texture and the flat 2D "
                 "picture-book cut-paper look of the starting frame; keep every character's face, hair, clothes and colours "
                 "exactly as in the starting frame; calm, steady camera like a multi-plane animation stand; never 3D, no new "
                 "characters, no text, captions or watermarks.")


class VideoError(RuntimeError):
    pass


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m["_dir"] = os.path.dirname(os.path.abspath(path))
    m.setdefault("model", "gemini-omni-1.2-flash")
    m.setdefault("resolution", "720p")
    m.setdefault("aspect_ratio", "16:9")
    m.setdefault("style", DEFAULT_STYLE)
    seen = set()
    for a in m.get("assets", []):
        if a.get("id") in seen or not a.get("id"):
            raise VideoError(f"asset id missing or repeated: {a.get('id')!r}")
        seen.add(a["id"])
        if not a.get("image") or not a.get("prompt", "").strip():
            raise VideoError(f"asset {a['id']!r} needs an image (start frame) and a prompt")
        d = int(a.get("duration", 8))
        top = 10 if a.get("last_image") else 15
        if not 3 <= d <= top:
            raise VideoError(f"asset {a['id']!r}: duration must be 3-{top} s")
    return m


def _path(m: dict, ref: str, out_dir: str) -> str:
    if ref.startswith("@"):
        return os.path.join(os.path.dirname(os.path.abspath(out_dir)), ref[1:])
    return os.path.join(m["_dir"], ref)


def _mime(path: str) -> str:
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def request(m: dict, a: dict, out_dir: str) -> dict:
    images = [_path(m, a["image"], out_dir)] + ([_path(m, a["last_image"], out_dir)] if a.get("last_image") else [])
    for p in images:
        if not os.path.exists(p):
            raise VideoError(f"asset {a['id']!r}: image not found: {p}")
    text = a["prompt"].strip() + "\n" + (a.get("style") or m["style"])
    return {"model": a.get("model", m["model"]), "images": images, "text": text, "duration": int(a.get("duration", 8)),
            "resolution": a.get("resolution", m["resolution"]), "aspect_ratio": a.get("aspect_ratio", m["aspect_ratio"])}


def key_of(req: dict) -> str:
    h = hashlib.sha1()
    for p in req["images"]:
        with open(p, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
    h.update(json.dumps({k: v for k, v in req.items() if k != "images"}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def estimate(req: dict) -> float:
    p = VIDEO_PRICE.get(req["model"], VIDEO_PRICE["gemini-omni-1.2-flash"])
    return req["duration"] * p["tokens_per_s"] * p["video_out"] / 1e6 + 2500 * p["in"] / 1e6


def _read_log(out_dir: str) -> dict:
    p = os.path.join(out_dir, "assets.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f).get("assets", [])}
    return {}


def _write_log(out_dir: str, records: dict, m: dict):
    total = round(sum(r.get("cost_usd", 0) for r in records.values()), 4)
    with open(os.path.join(out_dir, "assets.json"), "w", encoding="utf-8") as f:
        json.dump({"kind": "video", "model": m["model"], "total_cost_usd": total, "assets": list(records.values())},
                  f, ensure_ascii=False, indent=1)


def plan(m: dict, out_dir: str) -> list[dict]:
    done = _read_log(out_dir)
    steps = []
    for a in m["assets"]:
        req = request(m, a, out_dir)
        k = key_of(req)
        rec = done.get(a["id"])
        cached = bool(rec and rec.get("key") == k and os.path.exists(os.path.join(out_dir, rec["file"])))
        steps.append({"id": a["id"], "req": req, "key": k, "cached": cached})
    return steps


def _find_video(obj):
    """First {"type": "video", ...} block with data or a uri anywhere in an Interactions response."""
    if isinstance(obj, dict):
        if obj.get("type") == "video" and (obj.get("data") or obj.get("uri")):
            return obj
        for v in obj.values():
            hit = _find_video(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_video(v)
            if hit:
                return hit
    return None


def call_omni(req: dict, api_key: str, post, get=None, wait_s: int = 900) -> tuple[bytes, dict]:
    """One clip. -> (mp4 bytes, response without the video data)."""
    if get is None:
        import requests
        get = requests.get
    items = []
    for p in req["images"]:
        with open(p, "rb") as f:
            items.append({"type": "image", "mime_type": _mime(p), "data": base64.b64encode(f.read()).decode()})
    items.append({"type": "text", "text": req["text"]})
    body = {"model": req["model"], "input": items,
            "response_format": {"type": "video", "duration": f"{req['duration']}s", "resolution": req["resolution"],
                                "aspect_ratio": req["aspect_ratio"]}}
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last = ""
    for k in range(3):
        r = post(INTERACTIONS_API, headers=headers, json=body, timeout=900)
        if r.status_code == 200:
            data = r.json()
            break
        try:
            last = f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', '')}"
        except ValueError:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503, 504) and k < 2:
            time.sleep(15 * (k + 1))
            continue
        raise VideoError(last)
    else:
        raise VideoError(last)
    t0 = time.time()
    while not _find_video(data) and data.get("id") and data.get("status") in ("in_progress", "pending", "queued"):
        if time.time() - t0 > wait_s:
            raise VideoError("video generation did not finish in time")
        time.sleep(10)
        data = get(f"{INTERACTIONS_API}/{data['id']}", headers=headers, timeout=120).json()
    hit = _find_video(data)
    if not hit:
        raise VideoError(f"no video in the response (status: {data.get('status')})")
    if hit.get("data"):
        raw = base64.b64decode(hit["data"])
    else:  # delivered as a file: poll until it can be downloaded
        uri = hit["uri"]
        for _ in range(60):
            r = get(uri, headers={"x-goog-api-key": api_key}, timeout=300)
            if r.status_code == 200 and r.content[4:8] == b"ftyp":
                raw = r.content
                break
            time.sleep(10)
        else:
            raise VideoError(f"could not download the video from {uri}")
    meta = json.loads(json.dumps(data))
    blk = _find_video(meta)
    if blk and blk.get("data"):
        blk["data"] = f"<{len(raw)} bytes>"
    return raw, meta


def run(manifest_path: str, out_dir: str, max_calls: int = 4, dry_run: bool = False, api_key: str | None = None,
        post=None, get=None, log=print) -> dict:
    m = load_manifest(manifest_path)
    steps = plan(m, out_dir)
    new = [s for s in steps if not s["cached"]]
    log(f"[video] {m['model']} at {m['resolution']}: {len(steps)} clips, {len(new)} to generate, "
        f"{len(steps) - len(new)} already done")
    for s in steps:
        log(f"  {'cached' if s['cached'] else 'NEW':<6} {s['id']:<16} {s['req']['duration']:>2} s  from "
            f"{', '.join(os.path.basename(p) for p in s['req']['images'])}")
    est = sum(estimate(s["req"]) for s in new)
    log(f"[video] estimated cost of this run: ${est:.2f} (about $0.07 a second of 720p)")
    if len(new) > max_calls:
        raise VideoError(f"{len(new)} new clips exceeds the cap of {max_calls}; raise max_images to allow it")
    if dry_run or not new:
        return {"generated": [], "estimated_usd": est}
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise VideoError("GEMINI_API_KEY is not set")
    if post is None:
        import requests
        post = requests.post
    os.makedirs(out_dir, exist_ok=True)
    records = _read_log(out_dir)
    spent, made, failed = 0.0, [], []
    price = VIDEO_PRICE.get(m["model"], VIDEO_PRICE["gemini-omni-1.2-flash"])
    for s in new:
        t0 = time.time()
        try:
            raw, meta = call_omni(s["req"], api_key, post, get)
        except VideoError as e:
            log(f"  FAILED {s['id']}: {e}")
            failed.append(s["id"])
            continue
        u = meta.get("usage", {})
        vid_out = sum(x.get("tokens", 0) for x in u.get("output_tokens_by_modality", []) if x.get("modality") == "video")
        cost = (u.get("total_input_tokens", 0) * price["in"] + vid_out * price["video_out"]
                + max(0, u.get("total_output_tokens", 0) - vid_out + u.get("total_thought_tokens", 0)) * price["text_out"]) / 1e6
        if not u:
            cost = estimate(s["req"])
        fname = s["id"] + ".mp4"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(raw)
        spent += cost
        records[s["id"]] = {"id": s["id"], "type": "video", "key": s["key"], "file": fname, "model": s["req"]["model"],
                            "duration": s["req"]["duration"], "bytes": len(raw), "cost_usd": round(cost, 4),
                            "usage": u, "prompt": s["req"]["text"],
                            "images": [os.path.basename(p) for p in s["req"]["images"]],
                            "seconds": round(time.time() - t0, 1),
                            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_log(out_dir, records, m)
        made.append(s["id"])
        log(f"  made   {s['id']:<16} ${cost:.3f}  ({records[s['id']]['seconds']} s)")
    log(f"[video] generated {len(made)}, failed {len(failed)}, spent about ${spent:.3f}")
    if failed:
        raise VideoError(f"{len(failed)} clips failed: {', '.join(failed)}")
    return {"generated": made, "estimated_usd": est, "spent_usd": spent}
