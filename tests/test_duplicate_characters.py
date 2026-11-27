"""Cases for storyboard.duplicate_characters: which scenes show one character twice at once.

Run: python tests/test_duplicate_characters.py   (prints a table; exits 1 if any case disagrees)"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storyvid import pipeline as P, storyboard as SB  # noqa: E402

BASE = P.load_story(os.path.join(os.path.dirname(__file__), "..", "stories", "jingwei.json"))
BIRD = {"actor": "jingwei", "at": [700, 420], "scale": 0.6, "rig": "fly"}


def story_with(items, camera=None, actors=None):
    s = copy.deepcopy(BASE)
    sc = {"narration": "测试。", "subtitle_en": "Test.", "background": "dawn", "items": items}
    if camera:
        sc["camera"] = camera
    s["scenes"] = [sc]
    for name, spec in (actors or {}).items():
        s["actors"][name] = spec
    return s


def bird(**kw):
    b = copy.deepcopy(BIRD)
    b.update(kw)
    return b


variant = dict(copy.deepcopy(BASE["actors"]["jingwei"]), character="jingwei")
fish = {"parts": [{"id": "body", "shape": "ellipse", "rx": 60, "ry": 30, "paint": "#4a7fb0"}],
        "rigs": {"swim": [{"part": "body", "type": "bob", "amp": 4, "period": 1}]}}

CASES = [
    # name, story, expected: True = should be flagged
    ("two birds at once (old scene 9)", story_with([bird(), bird(at=[1300, 300], scale=0.4)]), True),
    ("same bird, one at a time (show windows)",
     story_with([bird(anims=[{"type": "show", "t": [0, 0.5]}]), bird(at=[1300, 300], anims=[{"type": "show", "t": [0.5, 1]}])]), False),
    ("dissolve between two poses (0.6 s crossfade)",
     story_with([bird(anims=[{"type": "fade", "to": 0, "t": [0.45, 0.55]}]),
                 bird(rig="glide", opacity=0.0, anims=[{"type": "fade", "to": 1, "t": [0.45, 0.55]}])]), False),
    ("copy off camera (zoomed in on the left)",
     story_with([bird(at=[500, 500]), bird(at=[1700, 500])], camera={"zoom": 2.0, "center": [480, 540]}), False),
    ("copy half in frame, origin outside", story_with([bird(), bird(at=[2090, 500], scale=1.4)]), True),
    ("reflection, declared", story_with([bird(at=[900, 500]), bird(at=[900, 700], scale=[0.6, -0.6], opacity=0.5, duplicate="reflection")]), False),
    ("two actors, one character (variant)", story_with([bird(), bird(actor="jingwei_carry", at=[1300, 300])], actors={"jingwei_carry": variant}), True),
    ("a flock of library gulls", story_with([{"actor": "bird_small", "at": [300 + 250 * k, 200 + 40 * k]} for k in range(5)]), False),
    ("character with only a swim rig, declared", story_with([{"actor": "fish", "at": [600, 600]}, {"actor": "fish", "at": [1200, 600]}],
                                                           actors={"fish": dict(fish, character=True)}), True),
    ("brief double: show windows overlap 4%",
     story_with([bird(anims=[{"type": "show", "t": [0, 0.52], "fade": 0.05}]),
                 bird(at=[1300, 300], anims=[{"type": "show", "t": [0.48, 1], "fade": 0.05}])]), True),
    ("a flock declared as a group actor", story_with([{"actor": "gulls", "at": [300 + 300 * k, 300]} for k in range(4)],
                                                    actors={"gulls": dict(copy.deepcopy(BASE["actors"]["jingwei"]), character=False)}), False),
]


def run(check):
    bad = 0
    print(f"{'case':<46} {'expected':<10} {'got':<10}")
    for name, s, want in CASES:
        got = bool(check(s))
        bad += got != want
        print(f"{name:<46} {'flag' if want else 'pass':<10} {'flag' if got else 'pass':<10} {'' if got == want else '  <-- wrong'}")
    for p in ("jingwei", "kuafu", "little_red_riding_hood", "tortoise_and_hare"):
        s = P.load_story(os.path.join(os.path.dirname(__file__), "..", "stories", p + ".json"))
        got = check(s)
        bad += bool(got)
        print(f"{'stories/' + p + '.json':<46} {'pass':<10} {'flag' if got else 'pass':<10} {'' if not got else '  <-- wrong: ' + got[0][:60]}")
    undeclared = copy.deepcopy(BASE)
    undeclared["actors"]["jingwei_carry"] = {k: v for k, v in BASE["actors"]["jingwei"].items() if k != "character"}
    got = SB.undeclared_actors(undeclared)
    bad += len(got) != 1
    print(f"{'actor without a character declaration':<46} {'error':<10} {'error' if got else 'none':<10}")
    print(f"{bad} wrong")
    return bad


if __name__ == "__main__":
    sys.exit(1 if run(lambda s: SB.check_characters(s)) else 0)
