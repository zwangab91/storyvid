"""Asset generation: an asset manifest -> images from an image model, generated once and cached.

    python -m storyvid.assets hybrid/assets/jingwei_step1.json --out generated/jingwei_step1 --dry-run
    GEMINI_API_KEY=... python -m storyvid.assets hybrid/assets/jingwei_step1.json --out generated/jingwei_step1

Manifest:
    {
      "model": "gemini-nano-banana-2.1",
      "image_size": "2K",                        # 1K | 2K | 4K
      "style": "appended to every prompt",
      "assets": [
        {"id": "bird", "aspect": "4:3", "prompt": "..."},
        {"id": "bird_parts", "aspect": "4:3", "prompt": "...", "refs": ["bird"]},
        {"id": "bird_b", "prompt": "...", "variant": 2}
      ]
    }

`refs` are earlier asset ids or image paths (relative to the manifest); they are sent as reference images so a
character keeps its look across poses and sheets. `variant` asks for another sample of an otherwise identical asset.

Each asset has a key: a hash of model, size, aspect, full prompt, variant and its references. An asset whose key
already has an image in the output folder is skipped, so re-running a manifest only pays for new or edited assets.
A run refuses to start if it would generate more than --max-images new images. --dry-run prints the plan and an
estimated cost without calling the API (no key needed).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time

# USD per 1M tokens for gemini-nano-banana-2.1 (Gemini API standard tier, as reported from Google's pricing page,
# Oct 2026) and the per-image list prices they imply. Used for estimates and the spend log only.
PRICE_PER_M = {"input": 1.50, "text_out": 7.50, "image_out": 30.00}
LIST_PRICE = {"1K": 0.0336, "2K": 0.0504, "4K": 0.113}
OVERHEAD = 0.25  # reference images, prompt and thinking tokens, on top of the image's list price (rough)

API = "https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent"
ASPECTS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
MAX_REF_SIDE = 2048


class AssetError(Exception):
    pass


# --------------------------------------------------------------------------- manifest and plan

def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m["_dir"] = os.path.dirname(os.path.abspath(path))
    m.setdefault("model", "gemini-nano-banana-2.1")
    m.setdefault("image_size", "2K")
    if m["image_size"] not in LIST_PRICE:
        raise AssetError(f"image_size must be one of {sorted(LIST_PRICE)}")
    seen = set()
    for a in m.get("assets", []):
        aid = a.get("id")
        if not aid or not str(aid).replace("_", "").replace("-", "").isalnum():
            raise AssetError(f"asset id {aid!r} must be letters, digits, '_' or '-'")
        if aid in seen:
            raise AssetError(f"duplicate asset id {aid!r}")
        if not a.get("prompt"):
            raise AssetError(f"asset {aid!r} has no prompt")
        if a.get("aspect", "16:9") not in ASPECTS:
            raise AssetError(f"asset {aid!r}: aspect must be one of {sorted(ASPECTS)}")
        for r in a.get("refs", []):
            if r not in seen and not os.path.isfile(os.path.join(m["_dir"], r)):
                raise AssetError(f"asset {aid!r}: ref {r!r} is neither an earlier asset id nor a file")
        seen.add(aid)
    if not seen:
        raise AssetError("manifest has no assets")
    return m


def full_prompt(m: dict, a: dict) -> str:
    style = m.get("style", "").strip()
    return a["prompt"].strip() + (("\n\n" + style) if style else "")


def plan(m: dict, out_dir: str) -> list[dict]:
    """-> one entry per asset: id, key, refs, cached (an image with this key already exists in out_dir)."""
    done = _read_log(out_dir)
    keys: dict[str, str] = {}
    steps = []
    for a in m["assets"]:
        ref_ids = []
        for r in a.get("refs", []):
            if r in keys:
                ref_ids.append("asset:" + keys[r])
            else:
                with open(os.path.join(m["_dir"], r), "rb") as f:
                    ref_ids.append("file:" + hashlib.sha1(f.read()).hexdigest())
        blob = json.dumps([m["model"], m["image_size"], a.get("aspect", "16:9"), full_prompt(m, a),
                           a.get("variant", 1), ref_ids], ensure_ascii=False)
        key = hashlib.sha1(blob.encode()).hexdigest()[:16]
        keys[a["id"]] = key
        rec = done.get(a["id"])
        cached = bool(rec and rec.get("key") == key and os.path.isfile(os.path.join(out_dir, rec["file"])))
        steps.append({"asset": a, "id": a["id"], "key": key, "cached": cached})
    return steps


def estimate(m: dict, steps: list[dict]) -> float:
    n = sum(not s["cached"] for s in steps)
    return n * LIST_PRICE[m["image_size"]] * (1 + OVERHEAD)


# --------------------------------------------------------------------------- spend log

def _log_path(out_dir):
    return os.path.join(out_dir, "assets.json")


def _read_log(out_dir) -> dict:
    p = _log_path(out_dir)
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f).get("assets", [])}


def _write_log(out_dir, records: dict, m: dict):
    os.makedirs(out_dir, exist_ok=True)
    order = [a["id"] for a in m["assets"]]
    rows = [records[i] for i in order if i in records] + [r for i, r in records.items() if i not in order]
    total = round(sum(r.get("cost_usd", 0.0) for r in rows), 4)
    tmp = _log_path(out_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": m["model"], "image_size": m["image_size"], "total_cost_usd": total, "assets": rows},
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, _log_path(out_dir))


def cost_from_usage(u: dict) -> float:
    """Approximate USD for one call from the API's usageMetadata."""
    img = sum(d.get("tokenCount", 0) for d in u.get("candidatesTokensDetails", []) if d.get("modality") == "IMAGE")
    out = u.get("candidatesTokenCount", 0)
    text = max(0, out - img) + u.get("thoughtsTokenCount", 0)
    return (u.get("promptTokenCount", 0) * PRICE_PER_M["input"] + text * PRICE_PER_M["text_out"]
            + img * PRICE_PER_M["image_out"]) / 1e6


# --------------------------------------------------------------------------- API

def _image_part(path: str) -> dict:
    from PIL import Image
    im = Image.open(path)
    if max(im.size) > MAX_REF_SIDE:
        k = MAX_REF_SIDE / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}


def call_gemini(model: str, parts: list[dict], aspect: str, size: str, api_key: str,
                post=None, attempts: int = 3) -> tuple[bytes, str, dict, str]:
    """One generateContent call. -> (image bytes, mime type, usageMetadata, any text the model returned)."""
    if post is None:
        import requests
        post = requests.post
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                 "imageConfig": {"aspectRatio": aspect, "imageSize": size}}}
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last = ""
    for ver in ("v1beta", "v1"):
        for k in range(attempts):
            r = post(API.format(ver=ver, model=model), headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                data = r.json()
                cand = (data.get("candidates") or [{}])[0]
                text = []
                for p in (cand.get("content") or {}).get("parts", []):
                    inline = p.get("inlineData") or p.get("inline_data")
                    if inline and inline.get("data"):
                        return (base64.b64decode(inline["data"]), inline.get("mimeType", "image/png"),
                                data.get("usageMetadata", {}), "\n".join(text))
                    if p.get("text"):
                        text.append(p["text"])
                reason = cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason")
                raise AssetError(f"no image returned (finish reason: {reason}); model said: {' '.join(text)[:300]}")
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                msg = r.text[:200]
            last = f"HTTP {r.status_code} on {ver}: {msg}"
            if r.status_code == 404:
                break  # try the other API version
            if r.status_code in (429, 500, 502, 503, 504) and k < attempts - 1:
                time.sleep(8 * (k + 1))
                continue
            raise AssetError(last)
    raise AssetError(last)


# --------------------------------------------------------------------------- run

def _save_png(data: bytes, path: str):
    from PIL import Image
    Image.open(io.BytesIO(data)).save(path, "PNG")


def contact_sheet(out_dir: str, ids: list[str], path: str, tw: int = 420):
    from PIL import Image, ImageDraw
    tiles = []
    for i in ids:
        p = os.path.join(out_dir, f"{i}.png")
        if os.path.isfile(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((tw, tw))
            tiles.append((i, im))
    if not tiles:
        return
    cols = min(4, len(tiles))
    rows = -(-len(tiles) // cols)
    th = max(t.height for _, t in tiles) + 26
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 10) + 10), "white")
    d = ImageDraw.Draw(sheet)
    for k, (i, im) in enumerate(tiles):
        x, y = 10 + (k % cols) * (tw + 10), 10 + (k // cols) * (th + 10)
        sheet.paste(im, (x, y))
        d.text((x, y + im.height + 6), i, fill=(40, 40, 40))
    sheet.save(path, quality=88)


def run(manifest_path: str, out_dir: str, max_images: int = 12, dry_run: bool = False,
        api_key: str | None = None, post=None, log=print) -> dict:
    m = load_manifest(manifest_path)
    steps = plan(m, out_dir)
    new = [s for s in steps if not s["cached"]]
    log(f"[assets] {m['model']} at {m['image_size']}: {len(steps)} assets, {len(new)} to generate, "
        f"{len(steps) - len(new)} already done")
    for s in steps:
        refs = ", ".join(s["asset"].get("refs", [])) or "-"
        log(f"  {'done' if s['cached'] else 'NEW '}  {s['id']:<24} {s['asset'].get('aspect', '16:9'):>5}  refs: {refs}")
    est = estimate(m, steps)
    log(f"[assets] estimated cost of this run: ${est:.2f} "
        f"(list price ${LIST_PRICE[m['image_size']]:.4f}/image + ~{int(OVERHEAD * 100)}% for inputs and thinking)")
    if len(new) > max_images:
        raise AssetError(f"{len(new)} new images exceeds the cap of {max_images}; raise --max-images to proceed")
    if dry_run or not new:
        return {"generated": [], "estimated_usd": est, "spent_usd": 0.0}
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AssetError("GEMINI_API_KEY is not set")
    os.makedirs(out_dir, exist_ok=True)
    records = _read_log(out_dir)
    paths = {i: os.path.join(out_dir, r["file"]) for i, r in records.items()}
    spent, made, failed = 0.0, [], []
    for s in steps:
        a = s["asset"]
        if s["cached"]:
            continue
        parts = []
        try:
            for r in a.get("refs", []):
                if r in paths:
                    parts.append(_image_part(paths[r]))
                elif any(x["id"] == r for x in m["assets"]):
                    raise AssetError(f"reference {r!r} was not generated")
                else:
                    parts.append(_image_part(os.path.join(m["_dir"], r)))
            parts.append({"text": full_prompt(m, a)})
            t0 = time.time()
            data, mime, usage, text = call_gemini(m["model"], parts, a.get("aspect", "16:9"), m["image_size"],
                                                  api_key, post=post)
        except AssetError as e:
            log(f"  FAILED {s['id']}: {e}")
            failed.append(s["id"])
            continue
        fname = f"{s['id']}.png"
        _save_png(data, os.path.join(out_dir, fname))
        cost = cost_from_usage(usage)
        spent += cost
        records[s["id"]] = {"id": s["id"], "key": s["key"], "file": fname, "aspect": a.get("aspect", "16:9"),
                            "refs": a.get("refs", []), "prompt": full_prompt(m, a), "model_text": text,
                            "usage": usage, "cost_usd": round(cost, 5), "seconds": round(time.time() - t0, 1),
                            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        paths[s["id"]] = os.path.join(out_dir, fname)
        _write_log(out_dir, records, m)  # after every image, so paid work is never lost
        made.append(s["id"])
        log(f"  made   {s['id']:<24} ${cost:.4f}  ({records[s['id']]['seconds']} s)")
    contact_sheet(out_dir, [a["id"] for a in m["assets"]], os.path.join(out_dir, "contact.jpg"))
    log(f"[assets] generated {len(made)}, failed {len(failed)}, spent about ${spent:.3f}")
    if failed:
        raise AssetError(f"{len(failed)} assets failed: {', '.join(failed)}")
    return {"generated": made, "estimated_usd": est, "spent_usd": spent}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", help="output folder (default: generated/<manifest name>)")
    ap.add_argument("--max-images", type=int, default=12, help="refuse to run if more new images are needed")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimated cost only")
    a = ap.parse_args(argv)
    out = a.out or os.path.join("generated", os.path.splitext(os.path.basename(a.manifest))[0])
    try:
        run(a.manifest, out, a.max_images, a.dry_run)
    except AssetError as e:
        print(f"[assets] error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
