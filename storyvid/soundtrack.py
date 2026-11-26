"""Narration and music from Google's models: storyboard -> audio manifest -> one narration clip per scene + a music clip.

    python -m storyvid.soundtrack plan stories/jingwei.json -o hybrid/assets/jingwei_audio.json
    python -m storyvid.assets hybrid/assets/jingwei_audio.json --out ../assets/jingwei_audio --dry-run

The second step is the same command (and the same "Generate assets" Actions workflow) as for images: storyvid.assets
hands manifests with "kind": "audio" to this module. To use the result, set "soundtrack" in the storyboard (or the
hybrid cast file) to the output folder; the renderer then voices every scene with its clip, times the captions from
the pauses in it, and lays the music under the film (looped, ducked under the voice). See audio.ClipNarrator.

Defaults copy Paperwing Studio's settings: Gemini 3.8 Flash-Lite TTS with the voice Kore and a storyteller direction,
one request per scene (the model plans the phrasing of the whole passage), and Lyria 3 Clip for a 30-second
instrumental loop.

Manifest:
    {
      "kind": "audio",
      "language": "zh",
      "speech": {"model": "gemini-3.8-flash-lite-tts", "voice": "Kore", "style": "Warm, gentle, ..."},
      "music": {"model": "lyria-3-clip-preview"},
      "assets": [
        {"id": "n00", "type": "speech", "text": "精卫填海"},
        {"id": "n01", "type": "speech", "text": "很久很久以前，..."},
        {"id": "music", "type": "music", "prompt": "Poetic gentle storybook acoustic chamber piece ..."}
      ]
    }

An asset may override "voice", "style" or "model". Keys hash everything that changes the result, so a re-run only
pays for new or edited lines, exactly as for images.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time

# USD, Gemini API standard tier. TTS: per 1M tokens; Gemini bills generated speech at 25 tokens per second of audio.
# Flash-Lite TTS is $0.50 in / $6.00 out through Dec 31, 2026 and $1.00 / $12.00 from Jan 1, 2027 (Google Cloud
# text-to-speech pricing page, Nov 2026). Lyria 3 Clip: $0.04 per 30-second clip. Used for estimates and the log.
TTS_PRICE = {"gemini-3.8-flash-lite-tts": (0.50, 6.00), "gemini-3.8-flash-tts": (0.50, 9.00)}
TTS_PRICE_2027 = {"gemini-3.8-flash-lite-tts": (1.00, 12.00), "gemini-3.8-flash-tts": (1.00, 18.00)}
MUSIC_PRICE = {"lyria-3-clip-preview": 0.04}
AUDIO_TOKENS_PER_S = 25
CHARS_PER_S = {"zh": 2.4, "en": 13.0}   # Gemini storyteller pace, measured on Paperwing's 精卫 narration (zh)

STYLE = {
    "zh": "Warm, gentle, expressive Mandarin Chinese children’s picture-book storyteller with unhurried pacing "
          "and poetic wonder",
    "en": "Warm, gentle, expressive children’s picture-book narrator in the classic bedtime film style, "
          "unhurried and clear",
}
MUSIC_SUFFIX = ("Strictly instrumental acoustic chamber arrangement, soft storybook dynamics suitable beneath voice "
                "narration, no singing or vocals.")

TTS_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
INTERACTIONS_API = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"


class SoundtrackError(Exception):
    pass


# --------------------------------------------------------------------------- plan a manifest from a storyboard

def scene_texts(story: dict) -> list[tuple[str, str]]:
    """(id, narration) for every narrated scene the renderer will voice, in order: title, scenes, end."""
    from .pipeline import build_scenes
    out = []
    for i, sc in enumerate(build_scenes(story)):
        text = sc.get("narration", "").strip()
        if text:
            out.append((f"n{i:02d}", text))
    return out


def plan_manifest(story: dict, voice: str = "Kore", style: str | None = None, model: str = "gemini-3.8-flash-lite-tts",
                  music_prompt: str | None = None, music_model: str = "lyria-3-clip-preview") -> dict:
    lang = "zh" if story.get("language", "en").startswith("zh") else "en"
    assets = [{"id": i, "type": "speech", "text": t} for i, t in scene_texts(story)]
    if music_prompt:
        assets.append({"id": "music", "type": "music", "prompt": music_prompt})
    return {"kind": "audio", "language": lang,
            "speech": {"model": model, "voice": voice, "style": style or STYLE[lang]},
            "music": {"model": music_model}, "assets": assets}


# --------------------------------------------------------------------------- manifest, keys, estimate

def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("kind") != "audio":
        raise SoundtrackError("not an audio manifest (\"kind\": \"audio\")")
    m.setdefault("language", "zh")
    sp = m.setdefault("speech", {})
    sp.setdefault("model", "gemini-3.8-flash-lite-tts")
    sp.setdefault("voice", "Kore")
    sp.setdefault("style", STYLE["zh" if m["language"].startswith("zh") else "en"])
    m.setdefault("music", {}).setdefault("model", "lyria-3-clip-preview")
    seen = set()
    for a in m.get("assets", []):
        aid = a.get("id")
        if not aid or not str(aid).replace("_", "").replace("-", "").isalnum():
            raise SoundtrackError(f"asset id {aid!r} must be letters, digits, '_' or '-'")
        if aid in seen:
            raise SoundtrackError(f"duplicate asset id {aid!r}")
        if a.get("type") == "speech" and not a.get("text", "").strip():
            raise SoundtrackError(f"speech asset {aid!r} has no text")
        if a.get("type") == "music" and not a.get("prompt", "").strip():
            raise SoundtrackError(f"music asset {aid!r} has no prompt")
        if a.get("type") not in ("speech", "music"):
            raise SoundtrackError(f"asset {aid!r}: type must be speech or music")
        seen.add(aid)
    if not seen:
        raise SoundtrackError("manifest has no assets")
    return m


def resolved(m: dict, a: dict) -> dict:
    """The full request an asset makes, with manifest defaults filled in."""
    if a["type"] == "speech":
        sp = m["speech"]
        return {"type": "speech", "model": a.get("model", sp["model"]), "voice": a.get("voice", sp["voice"]),
                "style": a.get("style", sp["style"]), "text": a["text"].strip()}
    prompt = a["prompt"].strip()
    if a.get("suffix", True) and MUSIC_SUFFIX not in prompt:
        prompt = prompt + " " + MUSIC_SUFFIX
    return {"type": "music", "model": a.get("model", m["music"]["model"]), "prompt": prompt,
            "variant": a.get("variant", 1)}


def key_of(req: dict) -> str:
    return hashlib.sha1(json.dumps(req, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _chars(text: str, lang: str) -> int:
    if lang.startswith("zh"):
        return len(re.findall(r"[㐀-鿿]", text)) or len(text)
    return len(text)


def estimate_one(req: dict, lang: str) -> float:
    if req["type"] == "music":
        return MUSIC_PRICE.get(req["model"], 0.08)
    p_in, p_out = TTS_PRICE.get(req["model"], (0.50, 9.00))
    secs = _chars(req["text"], lang) / CHARS_PER_S["zh" if lang.startswith("zh") else "en"]
    return (secs * AUDIO_TOKENS_PER_S * p_out + (len(req["text"]) + len(req["style"])) * p_in) / 1e6


def plan(m: dict, out_dir: str) -> list[dict]:
    done = _read_log(out_dir)
    steps = []
    for a in m["assets"]:
        req = resolved(m, a)
        key = key_of(req)
        rec = done.get(a["id"])
        cached = bool(rec and rec.get("key") == key and os.path.isfile(os.path.join(out_dir, rec["file"])))
        steps.append({"asset": a, "id": a["id"], "req": req, "key": key, "cached": cached})
    return steps


# --------------------------------------------------------------------------- log (same file name as images)

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
        json.dump({"kind": "audio", "language": m["language"], "speech": m["speech"], "total_cost_usd": total,
                   "assets": rows}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _log_path(out_dir))


# --------------------------------------------------------------------------- API calls

def _post_json(post, url, headers, body, attempts=3, timeout=300):
    last = ""
    for k in range(attempts):
        r = post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = r.text[:200]
        last = f"HTTP {r.status_code}: {msg}"
        if r.status_code in (429, 500, 502, 503, 504) and k < attempts - 1:
            time.sleep(8 * (k + 1))
            continue
        break
    raise SoundtrackError(last)


def call_tts(req: dict, api_key: str, post) -> tuple[bytes, str, dict]:
    """One narration clip. Same request Paperwing sends. -> (audio bytes, mime type, usageMetadata)."""
    body = {"contents": [{"role": "user", "parts": [{"text": req["text"], "speechMetadata": {"style": req["style"]}}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": req["voice"]}}}}}
    data = _post_json(post, TTS_API.format(model=req["model"]),
                      {"x-goog-api-key": api_key, "Content-Type": "application/json"}, body)
    cand = (data.get("candidates") or [{}])[0]
    for p in (cand.get("content") or {}).get("parts", []):
        inline = p.get("inlineData") or p.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"]), inline.get("mimeType", "audio/wav"), data.get("usageMetadata", {})
    reason = cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason")
    raise SoundtrackError(f"no audio returned (finish reason: {reason})")


def _find_audio(obj):
    """First {"type": "audio", "data": ...} block anywhere in an Interactions response."""
    if isinstance(obj, dict):
        if obj.get("type") == "audio" and obj.get("data"):
            return obj
        for v in obj.values():
            hit = _find_audio(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_audio(v)
            if hit:
                return hit
    return None


def call_music(req: dict, api_key: str, post, get=None, wait_s: int = 300) -> tuple[bytes, str, dict]:
    """One Lyria clip through the Interactions API. -> (audio bytes, mime type, response without the audio)."""
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json", "Api-Revision": API_REVISION}
    data = _post_json(post, INTERACTIONS_API, headers, {"model": req["model"], "input": req["prompt"]}, timeout=600)
    t0 = time.time()
    while not _find_audio(data) and data.get("id") and data.get("status") in ("in_progress", "pending", "queued"):
        if time.time() - t0 > wait_s:
            raise SoundtrackError("music generation did not finish in time")
        time.sleep(5)
        if get is None:
            import requests
            get = requests.get
        r = get(f"{INTERACTIONS_API}/{data['id']}", headers=headers, timeout=60)
        data = r.json()
    hit = _find_audio(data)
    if not hit:
        raise SoundtrackError(f"no audio in the response (status: {data.get('status')})")
    raw = base64.b64decode(hit["data"])
    meta = json.loads(json.dumps(data))
    blk = _find_audio(meta)
    if blk:
        blk["data"] = f"<{len(raw)} bytes>"
    return raw, hit.get("mime_type") or hit.get("mimeType") or "audio/mpeg", meta


# --------------------------------------------------------------------------- run

def _ext(mime: str) -> str:
    return {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3"}.get(mime, ".bin")


def run(manifest_path: str, out_dir: str, max_calls: int = 30, dry_run: bool = False,
        api_key: str | None = None, post=None, get=None, log=print) -> dict:
    m = load_manifest(manifest_path)
    steps = plan(m, out_dir)
    new = [s for s in steps if not s["cached"]]
    sp = m["speech"]
    log(f"[audio] narration {sp['model']} voice {sp['voice']}; music {m['music']['model']}: {len(steps)} items, "
        f"{len(new)} to generate, {len(steps) - len(new)} already done")
    for s in steps:
        what = s["req"].get("text") or s["req"].get("prompt")
        log(f"  {'done' if s['cached'] else 'NEW '}  {s['id']:<8} {s['req']['type']:<6} {what[:60]}")
    est = sum(estimate_one(s["req"], m["language"]) for s in new)
    log(f"[audio] estimated cost of this run: ${est:.3f} (narration at ~{AUDIO_TOKENS_PER_S} audio tokens/s, "
        f"music ${MUSIC_PRICE.get(m['music']['model'], 0.08):.2f}/clip)")
    if len(new) > max_calls:
        raise SoundtrackError(f"{len(new)} new items exceeds the cap of {max_calls}; raise --max-images to proceed")
    if dry_run or not new:
        return {"generated": [], "estimated_usd": est, "spent_usd": 0.0}
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SoundtrackError("GEMINI_API_KEY is not set")
    if post is None:
        import requests
        post = requests.post
    os.makedirs(out_dir, exist_ok=True)
    records = _read_log(out_dir)
    spent, made, failed = 0.0, [], []
    for s in steps:
        if s["cached"]:
            continue
        req = s["req"]
        t0 = time.time()
        try:
            if req["type"] == "speech":
                data, mime, usage = call_tts(req, api_key, post)
                p_in, p_out = TTS_PRICE.get(req["model"], (0.50, 9.00))
                cost = (usage.get("promptTokenCount", 0) * p_in + usage.get("candidatesTokenCount", 0) * p_out) / 1e6
                extra = {"usage": usage, "text": req["text"], "voice": req["voice"], "style": req["style"]}
            else:
                data, mime, meta = call_music(req, api_key, post, get)
                cost = MUSIC_PRICE.get(req["model"], 0.08)
                lyrics = [b.get("text") for st in meta.get("steps", []) for b in st.get("content", [])
                          if isinstance(b, dict) and b.get("type") == "text"]
                extra = {"prompt": req["prompt"], "model_text": "\n".join(x for x in lyrics if x)}
        except SoundtrackError as e:
            log(f"  FAILED {s['id']}: {e}")
            failed.append(s["id"])
            continue
        fname = s["id"] + _ext(mime)
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(data)
        spent += cost
        records[s["id"]] = dict({"id": s["id"], "type": req["type"], "key": s["key"], "file": fname, "mime": mime,
                                 "model": req["model"], "bytes": len(data), "cost_usd": round(cost, 5),
                                 "seconds": round(time.time() - t0, 1),
                                 "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, **extra)
        _write_log(out_dir, records, m)  # after every item, so paid work is never lost
        made.append(s["id"])
        log(f"  made   {s['id']:<8} ${cost:.4f}  ({records[s['id']]['seconds']} s)")
    log(f"[audio] generated {len(made)}, failed {len(failed)}, spent about ${spent:.3f}")
    if failed:
        raise SoundtrackError(f"{len(failed)} items failed: {', '.join(failed)}")
    return {"generated": made, "estimated_usd": est, "spent_usd": spent}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write an audio manifest for a storyboard")
    p.add_argument("storyboard")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--voice", default="Kore")
    p.add_argument("--style", help="speaking direction (default: Paperwing's storyteller line for the language)")
    p.add_argument("--model", default="gemini-3.8-flash-lite-tts")
    p.add_argument("--music-prompt", help="Lyria prompt; default: the storyboard's music.prompt, if any")
    a = ap.parse_args(argv)
    from .pipeline import load_story
    story = load_story(a.storyboard)
    music_prompt = a.music_prompt or (story.get("music") or {}).get("prompt")
    m = plan_manifest(story, a.voice, a.style, a.model, music_prompt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    lang = m["language"]
    est = sum(estimate_one(resolved(m, x), lang) for x in m["assets"])
    print(f"[audio] wrote {a.out}: {sum(x['type'] == 'speech' for x in m['assets'])} narration clips"
          f"{' + music' if music_prompt else ''}, about ${est:.3f}")


if __name__ == "__main__":
    try:
        main()
    except SoundtrackError as e:
        print(f"[audio] error: {e}", file=sys.stderr)
        sys.exit(1)
