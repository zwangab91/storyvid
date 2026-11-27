"""Story text -> storyboard JSON, using Claude.

Backends (picked automatically, or with --backend):
  claude-cli  : the `claude` CLI in print mode (runs on your Claude subscription)
  api         : Anthropic API (needs ANTHROPIC_API_KEY, `pip install anthropic`)

    python -m storyvid.storyboard story.txt --title "Little Red Riding Hood" --lang en -o stories/lrrh.json
    python -m storyvid.storyboard story.txt --lang zh --title 精卫填海 --subtitle "Jingwei Fills the Sea" -o stories/jw.json --critique 1

The draft is validated by actually building and rendering every scene at low resolution; errors are fed back to the
model for repair. With --critique N the contact sheet is shown to the model for N rounds of visual fixes.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .library import LIBRARY, library_doc
from .paper import PRESETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXAMPLE = os.path.join(ROOT, "stories", "tortoise_and_hare.json")


def spec_text() -> str:
    doc = open(os.path.join(ROOT, "docs", "STORYBOARD_SPEC.md"), encoding="utf-8").read()
    return doc.replace("PRESET_LIST", ", ".join(PRESETS)).replace("LIBRARY_LIST", library_doc())


def build_prompt(story_text: str, title: str, lang: str, subtitle: str | None, scenes: str, example_path: str) -> str:
    example = open(example_path, encoding="utf-8").read()
    zh = lang.startswith("zh")
    return f"""You are the art director and animator of a children's picture-book video series in a painted tissue-paper
collage style: bold hand-painted papers, simple cut shapes, white page backgrounds, gentle paper-puppet animation.
Turn the story below into a storyboard JSON for the storyvid renderer.

{spec_text()}

## A complete example storyboard (different story)
```json
{example}
```

## Your task
Title: {title}
{"Subtitle: " + subtitle if subtitle else ""}
Language of narration: {lang} {"(write the narration in simplified Chinese for young children, and give every scene a subtitle_en with the SAME number of sentences)" if zh else "(write the narration in warm, simple English for young children)"}
Scenes: {scenes}

Story to adapt (retell it in your own words; keep it faithful to the plot, gentle, and about 1.5-3 minutes when read aloud):
<story>
{story_text}
</story>

Design every recurring character once in "actors" (reuse the person template for humans), give them rigs, and keep
their look consistent across scenes. Use the built-in props for scenery. Check coordinates so characters stand on the
ground and stay inside the frame and above the caption band.

Reply with ONLY the JSON object, no commentary."""


# --------------------------------------------------------------------------- LLM backends

def _pick_backend(name: str | None) -> str:
    if name:
        return name
    if shutil.which("claude"):
        return "claude-cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    raise SystemExit("No LLM backend: install the Claude Code CLI (`claude`) or set ANTHROPIC_API_KEY.")


def ask_llm(prompt: str, backend: str, model: str | None, image: str | None = None) -> str:
    if backend == "claude-cli":
        cmd = ["claude", "-p", "--output-format", "text"]
        if model:
            cmd += ["--model", model]
        if image:
            cmd += ["--allowedTools", "Read"]
            prompt = f"First use the Read tool to look at the image {os.path.abspath(image)}.\n\n" + prompt
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {r.stderr[-2000:]}")
        return r.stdout
    if backend == "api":
        import anthropic
        client = anthropic.Anthropic()
        content = []
        if image:
            data = base64.b64encode(open(image, "rb").read()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(model=model or "claude-sonnet-5-5", max_tokens=32000,
                                     messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    raise ValueError(backend)


def extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


# --------------------------------------------------------------------------- validation

CHARACTER_RIGS = {"idle", "walk", "run", "fly", "talk"}


def character_of(name: str, spec: dict) -> str | None:
    """Who an actor depicts, or None for scenery and props.

    `"character": "<id>"` names it, so pose or costume variants of one character share an identity; `true` means the
    actor's own name; `false` means not a character (a flock, a grove). Without the key, a person or an actor with an
    idle, walk, run, fly or talk rig counts as a character of its own name."""
    c = spec.get("character")
    if isinstance(c, str) and c:
        return c
    if c is True:
        return name
    if c is False:
        return None
    return name if (spec.get("template") == "person" or CHARACTER_RIGS & set(spec.get("rigs", {}))) else None


def count_of(spec: dict) -> float:
    """How many of a character the story has: `"count"` is a number (3 little pigs) or "many" (a herd, a flock);
    default 1, an individual."""
    c = spec.get("count", 1)
    return float("inf") if c == "many" else float(c)


def bad_counts(story: dict) -> list[str]:
    errs, seen = [], {}
    for a, spec in story.get("actors", {}).items():
        c = spec.get("count", 1)
        if not (c == "many" or (isinstance(c, int) and not isinstance(c, bool) and c >= 1)):
            errs.append(f"actors.{a}: \"count\" must be a whole number of at least 1, or \"many\"")
            continue
        cid = character_of(a, spec)
        if cid:
            if cid in seen and seen[cid][1] != c:
                errs.append(f"actors.{a} and actors.{seen[cid][0]} depict {cid!r} but give different counts")
            seen.setdefault(cid, (a, c))
    return errs


def _coverage(fm, sc: dict, item: dict, u: float) -> int:
    """Pixels of the frame (at fm's size) where `item` alone shows at moment u, counted only while the item is at
    least 60% opaque. Opacity comes from the item's own animation state (fades, show windows), because overlapping
    paper parts of a half-faded figure stack above 60% in places. The pixels come from rendering the item on black
    and on white (alpha is the difference), so the camera, paths and hidden parts are exact."""
    from .anim import apply_anims, new_state
    st = new_state(*item.get("at", (0, 0)), scale=item.get("scale", 1.0), rot=item.get("rot", 0.0),
                   flip=item.get("flip", False), opacity=item.get("opacity", 1.0))
    st = apply_anims(st, item.get("anims", []), u, sc["_dur"], int(item.get("seed", 0)))
    if not st["vis"] or st["op"] < 0.6:
        return 0
    one = dict(sc)
    one["_items"] = [item]
    black = np.zeros((fm.H, fm.W, 3), np.float32)
    white = np.ones((fm.H, fm.W, 3), np.float32)
    fm.renderer.render(one, u, black)
    fm.renderer.render(one, u, white)
    return int(((1.0 - (white - black).mean(2)) > 0.3).sum())


def undeclared_actors(story: dict) -> list[str]:
    """Every story actor must say who it depicts, so two actors drawing one character can be told apart from two
    characters."""
    return [f"actors.{a}: say who it depicts with \"character\": a character id (pose or costume variants of one "
            f"character share it, e.g. \"wolf_in_bed\": {{\"character\": \"wolf\", ...}}; add \"count\" when the "
            f"story has several, e.g. 3 or \"many\" for a herd), or false for scenery and props"
            for a, spec in story.get("actors", {}).items() if "character" not in spec]


def check_characters(story: dict, scenes: list[dict] | None = None, fm=None, samples: int = 48,
                     min_px: int = 20) -> list[str]:
    """Moments when a character is on screen more times than the story has it: twice for an individual (two Jingwei
    birds, say), four times for "count": 3. "count": "many" (a herd, a flock) is never limited.

    Each item that depicts a character is rendered on its own at `samples` moments of the scene; a copy counts when it
    is at least 60% opaque and shows on at least `min_px` pixels of the 384x216 check frame. Any moment with two or more
    copies than the count is a problem. A dissolve between two poses passes, because during a crossfade the two never both reach
    60%; copies that take turns pass; an item marked `"duplicate": "<why>"` (a reflection, a memory) is left out."""
    from . import pipeline as P
    if scenes is None:
        scenes = P.build_scenes(story)
        for sc in scenes:
            sc["_dur"], sc["_captions"], sc["_start"] = 6.0, [], 0.0
    if fm is None:
        fm = P.FrameMaker(story, scenes, 384, 216, P.CACHE)
    who = {a: character_of(a, spec) for a, spec in story.get("actors", {}).items()}
    limit = {}
    for a, spec in story.get("actors", {}).items():
        if who[a]:
            limit[who[a]] = max(limit.get(who[a], 0), count_of(spec))
    errs = []
    for i, sc in enumerate(scenes):
        groups: dict[str, list] = {}
        for it in sc.get("_items", []):
            cid = who.get(it.get("actor"))
            if cid and not it.get("duplicate"):
                groups.setdefault(cid, []).append(it)
        for cid, its in groups.items():
            if len(its) <= limit[cid]:
                continue
            for k in range(samples):
                u = (k + 0.5) / samples
                on = sum(_coverage(fm, sc, it, u) >= min_px for it in its)
                if on > limit[cid]:
                    where = "title scene" if i == 0 else ("end scene" if i == len(scenes) - 1 else f"scenes[{i - 1}]")
                    actors = sorted({it["actor"] for it in its})
                    have = "one" if limit[cid] == 1 else f"{int(limit[cid])}"
                    errs.append(f"{where}: {cid!r} ({', '.join(actors)}) is on screen {on} times at once (around "
                                f"{u:.0%} of the scene) but the story has {have}. Show repeated action with one item "
                                f"(a path, part show windows); if the story has more, set \"count\" (a number or "
                                f"\"many\") on the actor; mark an intended copy \"duplicate\": \"<why>\".")
                    break
    return errs


def validate(story: dict, sheet_path: str | None = None) -> list[str]:
    """Build and render every scene at low res. Returns a list of problems (empty = OK)."""
    from . import pipeline as P
    from .audio import split_sentences
    errs: list[str] = []
    for key in ("title", "language", "scenes"):
        if key not in story:
            errs.append(f"missing top-level key {key!r}")
    if errs:
        return errs
    errs += undeclared_actors(story) + bad_counts(story)
    try:
        scenes = P.build_scenes(story)
    except Exception as e:  # noqa: BLE001
        return [f"building scenes failed: {type(e).__name__}: {e}"]
    for sc in scenes:
        sc["_dur"], sc["_captions"], sc["_start"] = 6.0, [], 0.0
    try:
        fm = P.FrameMaker(story, scenes, 384, 216, P.CACHE)
    except Exception as e:  # noqa: BLE001
        return [f"building actors failed: {type(e).__name__}: {e}"]
    for i, sc in enumerate(scenes):
        for u in (0.05, 0.5, 0.95):
            try:
                fm.scene_frame(i, 6.0 * u)
            except Exception as e:  # noqa: BLE001
                where = "title scene" if i == 0 else ("end scene" if i == len(scenes) - 1 else f"scenes[{i - 1}]")
                errs.append(f"{where} at t={u}: {type(e).__name__}: {e}")
                break
    errs += check_characters(story, scenes, fm)
    if story.get("language", "en").startswith("zh"):
        for i, sc in enumerate(story["scenes"]):
            a = split_sentences(sc.get("narration", ""), "zh")
            b = split_sentences(sc.get("subtitle_en", ""), "en")
            if sc.get("subtitle_en") and len(a) != len(b):
                errs.append(f"scenes[{i}]: narration has {len(a)} sentences but subtitle_en has {len(b)}; make them match")
    if not errs and sheet_path:
        P.contact_sheet(fm, scenes, sheet_path, tw=384)
    return errs


# --------------------------------------------------------------------------- main loop

def generate(story_text, title, lang="en", subtitle=None, scenes="10-14", backend=None, model=None,
             out_path=None, repair_rounds=3, critique=0, example=DEFAULT_EXAMPLE, log=print):
    backend = _pick_backend(backend)
    prompt = build_prompt(story_text, title, lang, subtitle, scenes, example)
    log(f"[storyboard] asking {backend} for a draft ...")
    reply = ask_llm(prompt, backend, model)
    story = extract_json(reply)
    tmpdir = tempfile.mkdtemp(prefix="storyvid_")
    for rnd in range(repair_rounds + 1):
        sheet = os.path.join(tmpdir, f"sheet_{rnd}.jpg")
        errs = validate(story, sheet if critique else None)
        if not errs:
            break
        log(f"[storyboard] round {rnd}: {len(errs)} problems -> repairing")
        if rnd == repair_rounds:
            raise RuntimeError("storyboard still invalid:\n" + "\n".join(errs))
        fix = (f"{spec_text()}\n\nThis storyboard fails validation:\n" + "\n".join(f"- {e}" for e in errs[:30]) +
               "\n\nReturn the corrected full storyboard JSON only.\n```json\n" + json.dumps(story, ensure_ascii=False) + "\n```")
        story = extract_json(ask_llm(fix, backend, model))
    for c in range(critique):
        sheet = os.path.join(tmpdir, f"critique_{c}.jpg")
        if validate(story, sheet):
            break
        log(f"[storyboard] critique round {c + 1}")
        ask = (f"{spec_text()}\n\nThe attached image is a contact sheet of the storyboard below (scene 0 = title card, "
               "last = end card; each tile is the scene at 60% of its duration, captions omitted). Critique it as a "
               "picture-book art director: characters off-ground or cut off, overlaps, unreadable shapes, empty or "
               "cluttered compositions, inconsistent characters, one character shown twice in a scene. Then return the improved full storyboard JSON only.\n"
               "```json\n" + json.dumps(story, ensure_ascii=False) + "\n```")
        try:
            cand = extract_json(ask_llm(ask, backend, model, image=sheet))
            if not validate(cand):
                story = cand
        except Exception as e:  # noqa: BLE001
            log(f"[storyboard] critique round failed: {e}")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(story, f, ensure_ascii=False, indent=1)
        log(f"[storyboard] wrote {out_path}")
    return story


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("story", help="text file with the story (or '-' for stdin)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--scenes", default="10-14")
    ap.add_argument("--backend", choices=["claude-cli", "api"])
    ap.add_argument("--model")
    ap.add_argument("--critique", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.story == "-" else open(a.story, encoding="utf-8").read()
    generate(text, a.title, a.lang, a.subtitle, a.scenes, a.backend, a.model, a.out, critique=a.critique)


if __name__ == "__main__":
    main()
