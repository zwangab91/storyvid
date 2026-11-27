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


def is_character(spec: dict) -> bool:
    """A story actor that is a character (shown once at a time): a person, or anything with a motion rig like idle,
    walk, run, fly or talk. `"character": true/false` on the actor overrides (false for, say, a flock of gulls)."""
    if "character" in spec:
        return bool(spec["character"])
    return spec.get("template") == "person" or bool(CHARACTER_RIGS & set(spec.get("rigs", {})))


def duplicate_characters(story: dict, scenes: list[dict]) -> list[str]:
    """Scenes that show one character twice at the same moment (two Jingwei birds, say). Copies that take turns are
    fine: one shown only in a time window, or one off screen while the other is on."""
    from .anim import apply_anims, new_state
    chars = {a for a, spec in story.get("actors", {}).items() if is_character(spec)}
    errs = []
    for i, sc in enumerate(scenes):
        by: dict[str, list] = {}
        for it in sc.get("items", []):
            if it.get("actor") in chars:
                by.setdefault(it["actor"], []).append(it)
        for actor, its in by.items():
            if len(its) < 2:
                continue
            for u in [k / 40 for k in range(1, 40)]:
                on = 0
                for it in its:
                    st = new_state(*it.get("at", (0, 0)), scale=it.get("scale", 1.0), opacity=it.get("opacity", 1.0))
                    st = apply_anims(st, it.get("anims", []), u, 6.0, int(it.get("seed", 0)))
                    on += st["vis"] and st["op"] > 0.05 and -150 < st["x"] < 2070 and -150 < st["y"] < 1230
                if on > 1:
                    where = "title scene" if i == 0 else ("end scene" if i == len(scenes) - 1 else f"scenes[{i - 1}]")
                    errs.append(f"{where}: the character {actor!r} is on screen {on} times at once (at {u:.0%} of the "
                                f"scene); show a character once, and show repeated action with one item (a path, "
                                f"part show windows) - or mark the actor \"character\": false if copies are meant")
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
    errs += duplicate_characters(story, scenes)
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
               "cluttered compositions, inconsistent characters. Then return the improved full storyboard JSON only.\n"
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
