"""Built-in prop actors any storyboard can use (scenery, sky, plants, water).

Each actor's origin (0,0) is documented; place it with "at" and resize with "scale".
Story-specific characters are defined inside the storyboard's "actors" block.
"""
from __future__ import annotations

import math


def _petals(c, n, r_len, r_w, paint, start_id="petal", rot0=0.0, parent=None):
    parts = []
    for i in range(n):
        a = rot0 + 360 * i / n
        ar = math.radians(a)
        cx = c[0] + math.cos(ar) * r_len * 0.9
        cy = c[1] + math.sin(ar) * r_len * 0.9
        p = {"id": f"{start_id}{i}", "shape": "ellipse", "c": [cx, cy], "r": [r_len, r_w], "rot": a,
             "paint": paint, "shadow": 0.6}
        if parent:
            p["parent"] = parent
        parts.append(p)
    return parts


def _flower(paint, center="flower_yellow", stem_h=160):
    head = [0, -stem_h]
    parts = [{"id": "stem", "shape": "stroke", "points": [[0, 0], [6, -stem_h * 0.5], [0, -stem_h]],
              "width": [12, 8], "paint": "grass", "pivot": [0, 0]},
             {"id": "leaf", "shape": "ellipse", "c": [24, -stem_h * 0.35], "r": [30, 11], "rot": -30,
              "paint": "leaf", "parent": "stem"}]
    parts += _petals(head, 6, 34, 18, paint, parent="stem")
    parts.append({"id": "center", "shape": "circle", "c": head, "r": 20, "paint": center, "parent": "stem"})
    return {"parts": parts, "rigs": {"sway": [{"part": "stem", "type": "swing", "deg": 4, "period": 2.6}]}}


def _tree_round(crown="leaf", crown2="dark_leaf", trunk="bark", h=420, seed=0):
    return {"parts": [
        {"id": "trunk", "shape": "poly", "smooth": True, "corners": [0, 5],
         "points": [[-34, 0], [-22, -h * 0.35], [-26, -h * 0.62], [24, -h * 0.62], [20, -h * 0.35], [34, 0]],
         "paint": trunk},
        {"id": "crown_b", "shape": "blob", "c": [-70, -h * 0.72], "r": [150, 120], "paint": crown2,
         "shape_seed": seed + 1, "pivot": [0, -h * 0.55]},
        {"id": "crown", "shape": "blob", "c": [40, -h * 0.8], "r": [170, 140], "paint": crown,
         "shape_seed": seed + 2, "pivot": [0, -h * 0.55]},
    ], "rigs": {"sway": [{"part": ["crown", "crown_b"], "type": "swing", "deg": 2.0, "period": 3.4}]}}


LIBRARY: dict[str, dict] = {
    # ---------------------------------------------------------------- sky
    "sun": {  # origin: centre; radius ~170
        "parts": [
            {"id": "rays", "shape": "star", "c": [0, 0], "r": [112, 172], "n": 14, "paint": "sun_fire", "pivot": [0, 0]},
            {"id": "disc", "shape": "circle", "c": [0, 0], "r": 112, "paint": "sun"},
        ],
        "rigs": {"glow": [{"part": "rays", "type": "spin", "speed": 6},
                          {"part": "disc", "type": "pulse", "amp": 0.025, "period": 2.4}]}},
    "moon": {  # origin: centre; crescent radius 90
        "parts": [{"id": "moon", "shape": "circle", "c": [0, 0], "r": 90, "paint": "moon",
                   "cut": [{"shape": "circle", "c": [42, -26], "r": 80}]}]},
    "full_moon": {"parts": [{"id": "moon", "shape": "circle", "c": [0, 0], "r": 90, "paint": "moon"}]},
    "cloud": {  # origin: centre; ~360 x 170
        "parts": [{"id": "cloud", "shape": "cloud", "c": [0, 0], "r": [180, 95], "paint": "cloud", "puffs": 5}]},
    "cloud_long": {"parts": [{"id": "cloud", "shape": "cloud", "c": [0, 0], "r": [260, 90], "paint": "cloud",
                              "puffs": 7, "shape_seed": 3}]},
    "star": {  # origin: centre; twinkles with rig "twinkle"
        "parts": [{"id": "star", "shape": "star", "c": [0, 0], "r": [11, 26], "n": 5, "paint": "gold", "shadow": 0.5}],
        "rigs": {"twinkle": [{"part": "star", "type": "pulse", "amp": 0.18, "period": 1.7}]}},
    "bird_small": {  # distant bird, origin centre; rig "flap"
        "parts": [
            {"id": "wl", "shape": "stroke", "points": [[0, 0], [-22, -14], [-44, -4]], "width": [7, 3],
             "paint": "black", "pivot": [0, 0], "shadow": 0.3},
            {"id": "wr", "shape": "stroke", "points": [[0, 0], [22, -14], [44, -4]], "width": [7, 3],
             "paint": "black", "pivot": [0, 0], "shadow": 0.3}],
        "rigs": {"flap": [{"part": "wl", "type": "swing", "deg": 22, "period": 0.5},
                          {"part": "wr", "type": "swing", "deg": -22, "period": 0.5}]}},
    # ---------------------------------------------------------------- land
    "hill": {  # origin: bottom centre; 1400 wide, 330 high
        "parts": [{"id": "hill", "shape": "poly", "smooth": True, "corners": [0, 4],
                   "points": [[-700, 0], [-480, -210], [0, -330], [480, -230], [700, 0]], "paint": "meadow_far"}]},
    "ground": {  # origin: top-left of a full-width meadow strip at y; use at:[0, y]
        "parts": [{"id": "ground", "shape": "wave", "x0": -260, "x1": 2180, "y": 0, "amp": 22, "wavelength": 760,
                   "bottom": 700, "paint": "grass", "edge": "cut"}]},
    "mountain": {  # origin: bottom centre; 900 wide, 620 high
        "parts": [
            {"id": "m", "shape": "poly", "smooth": True, "corners": [0, 6],
             "points": [[-450, 0], [-250, -330], [-90, -560], [0, -620], [110, -540], [280, -300], [450, 0]],
             "paint": "mountain"},
            {"id": "shade", "shape": "poly", "smooth": True, "corners": [0, 3],
             "points": [[0, -620], [110, -540], [280, -300], [450, 0], [120, 0], [60, -300]],
             "paint": {"preset": "mountain", "colors": ["#4f5190", "#6567a8", "#3b3d74"]}, "shadow": 0.3}]},
    "rock": {"parts": [{"id": "rock", "shape": "blob", "c": [0, -45], "r": [80, 50], "paint": "stone",
                        "shape_seed": 4, "jag": 0.12}]},
    "pebble": {"parts": [{"id": "p", "shape": "blob", "c": [0, 0], "r": [16, 12], "paint": "stone", "jag": 0.12,
                          "shape_seed": 2}]},
    "twig": {"parts": [
        {"id": "stick", "shape": "stroke", "points": [[-40, 6], [0, 0], [42, -6]], "width": [7, 5], "paint": "bark"},
        {"id": "leaf", "shape": "ellipse", "c": [22, -14], "r": [14, 7], "rot": -40, "paint": "leaf", "shadow": 0.4}]},
    "grass_tuft": {  # origin: base
        "parts": [{"id": f"b{i}", "shape": "stroke",
                   "points": [[x, 0], [x + dx * 0.4, -h * 0.55], [x + dx, -h]], "width": [14, 2],
                   "paint": "grass", "pivot": [x, 0], "shadow": 0.5}
                  for i, (x, dx, h) in enumerate([(-24, -26, 70), (-10, -8, 100), (0, 10, 120), (12, 26, 92), (24, 40, 64)])],
        "rigs": {"sway": [{"part": [f"b{i}" for i in range(5)], "type": "swing", "deg": 5, "period": 2.2}]}},
    "flower_red": _flower("flower_red"),
    "flower_yellow": _flower("flower_yellow", center="flower_red", stem_h=140),
    "flower_purple": _flower("flower_purple", stem_h=180),
    "tree": _tree_round(),
    "tree_dark": _tree_round("dark_leaf", "leaf", seed=10),
    "peach_tree": _tree_round("peach_blossom", "leaf", seed=20),
    "pine": {  # origin: base; 520 high
        "parts": [
            {"id": "trunk", "shape": "rect", "c": [0, -50], "size": [44, 110], "paint": "bark"},
            {"id": "t1", "shape": "poly", "smooth": True, "corners": [0, 1, 2], "points": [[-170, -80], [170, -80], [0, -300]], "paint": "dark_leaf", "pivot": [0, -80]},
            {"id": "t2", "shape": "poly", "smooth": True, "corners": [0, 1, 2], "points": [[-140, -200], [140, -200], [0, -420]], "paint": "dark_leaf", "pivot": [0, -200]},
            {"id": "t3", "shape": "poly", "smooth": True, "corners": [0, 1, 2], "points": [[-100, -320], [100, -320], [0, -520]], "paint": "leaf", "pivot": [0, -320]}],
        "rigs": {"sway": [{"part": ["t1", "t2", "t3"], "type": "swing", "deg": 1.5, "period": 3.0}]}},
    "bush": {"parts": [
        {"id": "b1", "shape": "blob", "c": [-60, -60], "r": [90, 70], "paint": "dark_leaf", "shape_seed": 5},
        {"id": "b2", "shape": "blob", "c": [50, -70], "r": [100, 80], "paint": "leaf", "shape_seed": 6}]},
    # ---------------------------------------------------------------- water
    "sea": {  # origin: (0,0) = left end of the water line; spans x 0..1920 (+margins), down to y+700
        "parts": [
            {"id": "w1", "shape": "wave", "x0": -300, "x1": 2220, "y": 0, "amp": 22, "wavelength": 420, "bottom": 760,
             "paint": "deep_sea", "shape_seed": 1, "pivot": [960, 0]},
            {"id": "w2", "shape": "wave", "x0": -300, "x1": 2220, "y": 90, "amp": 26, "wavelength": 520, "bottom": 760,
             "paint": "sea", "shape_seed": 2, "phase": 1.3, "pivot": [960, 90]},
            {"id": "w3", "shape": "wave", "x0": -300, "x1": 2220, "y": 210, "amp": 30, "wavelength": 610, "bottom": 760,
             "paint": "water", "shape_seed": 3, "phase": 2.6, "pivot": [960, 210]}],
        "rigs": {"waves": [
            {"part": "w1", "type": "bob", "amp": 40, "dir": [1, 0.15], "period": 5.5},
            {"part": "w2", "type": "bob", "amp": 55, "dir": [-1, 0.2], "period": 4.6},
            {"part": "w3", "type": "bob", "amp": 45, "dir": [1, 0.25], "period": 3.9}]}},
    "river": {  # origin: centre; a winding horizontal band 2400 wide, ~120 thick
        "parts": [{"id": "river", "shape": "stroke", "points": [[-1200, 20], [-600, -30], [0, 25], [600, -20], [1200, 15]],
                   "width": 120, "paint": "river", "cap": "flat", "shadow": 0.3}]},
    "splash": {  # origin: centre; white water burst, use with pop
        "parts": [{"id": "s", "shape": "star", "c": [0, 0], "r": [30, 70], "n": 9, "smooth": True, "paint": "foam"}]},
    # ---------------------------------------------------------------- buildings
    "cottage": {  # origin: bottom centre; 460 wide, 470 high
        "parts": [
            {"id": "chimney", "shape": "rect", "c": [120, -370], "size": [50, 110], "paint": "brick"},
            {"id": "walls", "shape": "rect", "c": [0, -130], "size": [380, 260], "paint": "cream"},
            {"id": "roof", "shape": "poly", "smooth": True, "corners": [0, 1, 2],
             "points": [[-240, -250], [240, -250], [0, -450]], "paint": "roof"},
            {"id": "door", "shape": "rect", "c": [-80, -80], "size": [90, 160], "radius": 40, "paint": "wood"},
            {"id": "knob", "shape": "circle", "c": [-52, -80], "r": 7, "paint": "gold", "shadow": 0.4},
            {"id": "window", "shape": "rect", "c": [90, -150], "size": [110, 90], "radius": 8, "paint": "sky"},
            {"id": "sill", "shape": "rect", "c": [90, -100], "size": [130, 16], "paint": "wood"}]},
}


def library_doc() -> str:
    """One line per prop for the LLM storyboard prompt."""
    notes = {
        "sun": "rig glow; origin centre, r~170", "moon": "crescent, origin centre", "full_moon": "origin centre",
        "cloud": "origin centre ~360x170; use anims drift", "cloud_long": "origin centre ~520x180",
        "star": "rig twinkle; small", "bird_small": "distant bird, rig flap",
        "hill": "origin bottom-centre, 1400x330", "ground": "full-width meadow, place at [0,y]",
        "mountain": "origin bottom-centre 900x620", "rock": "origin bottom", "pebble": "origin centre, tiny",
        "twig": "origin centre, small stick with leaf", "grass_tuft": "origin base, rig sway",
        "flower_red": "origin base, rig sway", "flower_yellow": "origin base, rig sway",
        "flower_purple": "origin base, rig sway", "tree": "origin base, 420 tall, rig sway",
        "tree_dark": "origin base, rig sway", "peach_tree": "blossoming tree, origin base, rig sway",
        "pine": "origin base, 520 tall, rig sway", "bush": "origin base",
        "sea": "place at [0,y_waterline]; 3 wave bands to bottom; rig waves",
        "river": "origin centre, winding 2400-wide band", "splash": "white burst, use pop",
        "cottage": "origin bottom-centre 460x470",
    }
    return "\n".join(f"- {k}: {notes.get(k, '')}" for k in LIBRARY)


# --------------------------------------------------------------------------- templates

def person(height=400, child=False, skin="skin_light", hair="hair_brown", hair_style="short", top="blue_cloth",
           bottom=None, dress=True, apron=None, hood=None, hat=None, hat_paint="white", beard=None, glasses=False,
           shawl=None, shoes="black", holding=None, facing=0.0):
    """Front-facing paper-doll person. Origin: between the feet. Returns an actor spec.

    hair_style: short | long | bun | bob | bald ; hat: nightcap | cap | straw | None ;
    hood: paint for a riding hood/cape ; holding: basket | axe | None (in the right hand).
    facing: -1..1 turns the face slightly left/right (shifts features).
    """
    k = height / (330.0 if child else 400.0)
    if child:
        head_c, head_r, sh_y, waist_y, hip_y, hand_y, sh_x = (0, -250), 44, -200, -130, -70, -120, 30
    else:
        head_c, head_r, sh_y, waist_y, hip_y, hand_y, sh_x = (0, -305), 42, -255, -170, -115, -150, 36
    fx = facing * 10

    def P(x, y):
        return [round(x * k, 1), round(y * k, 1)]

    def PL(pts):
        return [P(*p) for p in pts]
    hx, hy = head_c
    parts = []
    # cape / hood behind everything
    if hood:
        parts.append({"id": "cape", "shape": "poly", "smooth": True, "corners": [0, 8],
                      "points": PL([(-sh_x * 2.3, hip_y - 10), (-sh_x * 1.8, sh_y + 20), (-head_r * 1.25, hy + 8),
                                    (-head_r * 0.9, hy - head_r * 1.05), (0, hy - head_r * 1.4), (head_r * 0.9, hy - head_r * 1.05),
                                    (head_r * 1.25, hy + 8), (sh_x * 1.8, sh_y + 20), (sh_x * 2.3, hip_y - 10)]),
                      "paint": hood, "pivot": P(0, sh_y)})
    if hair_style == "long":
        parts.append({"id": "hair_back", "shape": "poly", "smooth": True,
                      "points": PL([(-head_r * 1.1, hy - 10), (-head_r * 1.15, hy + head_r * 1.5), (0, hy + head_r * 1.3),
                                    (head_r * 1.15, hy + head_r * 1.5), (head_r * 1.1, hy - 10), (0, hy - head_r * 1.12)]),
                      "paint": hair, "parent": "head"})
    # legs
    for side, sx in (("l", -1), ("r", 1)):
        parts.append({"id": f"leg_{side}", "shape": "stroke", "points": PL([(sx * 16, hip_y - 15), (sx * 17, -8)]),
                      "width": 17 * k, "paint": bottom if (bottom and not dress) else skin, "pivot": P(sx * 16, hip_y - 15)})
        parts.append({"id": f"shoe_{side}", "shape": "ellipse", "c": P(sx * 20, -6), "r": [16 * k, 8 * k],
                      "paint": shoes, "parent": f"leg_{side}"})
    # body
    if dress:
        parts.append({"id": "body", "shape": "poly", "smooth": True, "corners": [0, 3],
                      "points": PL([(-sh_x * 1.75, hip_y + 18), (-sh_x * 0.95, sh_y), (sh_x * 0.95, sh_y), (sh_x * 1.75, hip_y + 18)]),
                      "paint": top, "pivot": P(0, hip_y)})
    else:
        parts.append({"id": "pants", "shape": "rect", "c": P(0, (waist_y + hip_y) / 2 + 10), "size": [sh_x * 2.0 * k, (hip_y - waist_y + 40) * k],
                      "radius": 8 * k, "paint": bottom or "blue_cloth"})
        parts.append({"id": "body", "shape": "rect", "c": P(0, (sh_y + waist_y) / 2 + 8), "size": [sh_x * 2.3 * k, (waist_y - sh_y + 40) * k],
                      "radius": 18 * k, "paint": top, "pivot": P(0, waist_y)})
    if apron:
        parts.append({"id": "apron", "shape": "rect", "c": P(0, (sh_y + hip_y) / 2 + 22), "size": [sh_x * 1.3 * k, (hip_y - sh_y) * 0.75 * k],
                      "radius": 10 * k, "paint": apron, "parent": "body", "shadow": 0.6})
    if shawl:
        parts.append({"id": "shawl", "shape": "poly", "smooth": True, "corners": [0, 2],
                      "points": PL([(-sh_x * 1.25, sh_y - 6), (0, sh_y + 70), (sh_x * 1.25, sh_y - 6), (0, sh_y + 10)]),
                      "paint": shawl, "parent": "body"})
    # arms (sleeves) + hands
    sleeve = hood if hood else top
    for side, sx in (("l", -1), ("r", 1)):
        parts.append({"id": f"arm_{side}", "shape": "stroke", "points": PL([(sx * sh_x, sh_y + 8), (sx * (sh_x + 14), (sh_y + hand_y) / 2), (sx * (sh_x + 22), hand_y)]),
                      "width": [17 * k, 14 * k], "paint": sleeve, "pivot": P(sx * sh_x, sh_y + 8)})
        parts.append({"id": f"hand_{side}", "shape": "circle", "c": P(sx * (sh_x + 23), hand_y + 6), "r": 9 * k,
                      "paint": skin, "parent": f"arm_{side}"})
    # held items (right hand)
    if holding == "basket":
        bx, by = sh_x + 23, hand_y + 6
        parts += [
            {"id": "basket_handle", "shape": "stroke", "points": PL([(bx - 22, by + 22), (bx, by - 8), (bx + 22, by + 22)]),
             "width": 6 * k, "paint": "wicker", "parent": "arm_r", "shadow": 0.5},
            {"id": "basket", "shape": "poly", "smooth": True, "corners": [0, 1],
             "points": PL([(bx - 34, by + 18), (bx + 34, by + 18), (bx + 26, by + 56), (bx - 26, by + 56)]),
             "paint": "wicker", "parent": "arm_r"},
            {"id": "basket_cloth", "shape": "ellipse", "c": P(bx, by + 17), "r": [30 * k, 9 * k],
             "paint": {"colors": ["#ffffff", "#d7262e"], "style": "stripes", "period": 10, "angle": 45}, "parent": "arm_r", "shadow": 0.4}]
        parts.append(parts.pop(next(i for i, p in enumerate(parts) if p["id"] == "hand_r")))
    if holding == "axe":
        bx, by = sh_x + 23, hand_y + 6
        parts += [
            {"id": "axe_handle", "shape": "stroke", "points": PL([(bx, by + 50), (bx + 4, by - 150)]), "width": 10 * k,
             "paint": "wood", "parent": "arm_r"},
            {"id": "axe_blade", "shape": "poly", "smooth": True, "corners": [0, 3],
             "points": PL([(bx + 2, by - 150), (bx + 50, by - 168), (bx + 58, by - 115), (bx + 4, by - 122)]),
             "paint": "stone", "parent": "arm_r"}]
        parts.append(parts.pop(next(i for i, p in enumerate(parts) if p["id"] == "hand_r")))
    if holding == "staff":
        bx, by = sh_x + 23, hand_y + 6
        parts += [
            {"id": "staff", "shape": "stroke", "points": PL([(bx - 2, by + 145), (bx + 2, by), (bx + 8, by - 210)]),
             "width": [13 * k, 10 * k], "paint": "bark", "parent": "arm_r"},
            {"id": "staff_knob", "shape": "blob", "c": P(bx + 8, by - 215), "r": [15 * k, 13 * k], "paint": "bark",
             "parent": "arm_r", "shape_seed": 2}]
        parts.append(parts.pop(next(i for i, p in enumerate(parts) if p["id"] == "hand_r")))
    # head
    parts.append({"id": "neck", "shape": "rect", "c": P(0, sh_y - 8), "size": [18 * k, 26 * k], "paint": skin, "shadow": 0.3})
    parts.append({"id": "head", "shape": "circle", "c": P(hx, hy), "r": head_r * k, "paint": skin, "pivot": P(0, sh_y - 4)})
    ex = 14 + (2 if child else 0)
    parts += [
        {"id": "eye_l", "shape": "circle", "c": P(hx - ex + fx, hy - 2), "r": 5.2 * k, "paint": "black", "parent": "head", "shadow": 0.2},
        {"id": "eye_r", "shape": "circle", "c": P(hx + ex + fx, hy - 2), "r": 5.2 * k, "paint": "black", "parent": "head", "shadow": 0.2},
        {"id": "cheek_l", "shape": "circle", "c": P(hx - 25 + fx, hy + 13), "r": 7 * k, "paint": "pink", "parent": "head", "shadow": 0, "opacity": 0.75},
        {"id": "cheek_r", "shape": "circle", "c": P(hx + 25 + fx, hy + 13), "r": 7 * k, "paint": "pink", "parent": "head", "shadow": 0, "opacity": 0.75},
        {"id": "mouth", "shape": "stroke", "points": PL([(hx - 9 + fx, hy + 18), (hx + fx, hy + 23), (hx + 9 + fx, hy + 18)]),
         "width": 3.2 * k, "paint": "#8a2a2a", "parent": "head", "shadow": 0},
        {"id": "mouth_open", "shape": "ellipse", "c": P(hx + fx, hy + 22), "r": [9 * k, 8 * k], "paint": "#7a1f2b",
         "parent": "head", "hidden": True, "shadow": 0.2},
    ]
    # hair front
    if hair_style in ("short", "long", "bob", "bun"):
        parts.append({"id": "hair", "shape": "poly", "smooth": True,
                      "points": PL([(hx - head_r * 1.02, hy + 2), (hx - head_r * 0.85, hy - head_r * 0.7), (hx, hy - head_r * 1.1),
                                    (hx + head_r * 0.85, hy - head_r * 0.7), (hx + head_r * 1.02, hy + 2),
                                    (hx + head_r * 0.6, hy - head_r * 0.45), (hx + fx, hy - head_r * 0.6), (hx - head_r * 0.6, hy - head_r * 0.45)]),
                      "paint": hair, "parent": "head"})
    if hair_style == "bun":
        parts.append({"id": "bun", "shape": "circle", "c": P(hx, hy - head_r * 1.15), "r": head_r * 0.45 * k, "paint": hair, "parent": "head"})
    if hair_style == "bob":
        for sx in (-1, 1):
            parts.append({"id": f"bob_{'l' if sx < 0 else 'r'}", "shape": "ellipse", "c": P(hx + sx * head_r * 0.95, hy + 6),
                          "r": [head_r * 0.32 * k, head_r * 0.62 * k], "paint": hair, "parent": "head"})
    if beard:
        parts += [
            {"id": "beard", "shape": "blob", "c": P(hx, hy + head_r * 0.68), "r": [head_r * 0.82 * k, head_r * 0.5 * k],
             "paint": beard, "parent": "head", "shape_seed": 3, "jag": 0.1},
            {"id": "moustache", "shape": "ellipse", "c": P(hx + fx, hy + 14), "r": [19 * k, 6 * k], "paint": beard, "parent": "head"}]
        # mouth sits on top of the beard
        for mid in ("mouth", "mouth_open"):
            parts.append(parts.pop(next(i for i, p in enumerate(parts) if p["id"] == mid)))
    if glasses:
        for sx in (-1, 1):
            parts.append({"id": f"glass_{'l' if sx < 0 else 'r'}", "shape": "circle", "c": P(hx + sx * ex + fx, hy - 2),
                          "r": 12 * k, "paint": "black", "parent": "head", "shadow": 0,
                          "cut": [{"shape": "circle", "c": P(hx + sx * ex + fx, hy - 2), "r": 9.3 * k}]})
        parts.append({"id": "glass_bridge", "shape": "stroke", "points": PL([(hx - ex + 11 + fx, hy - 3), (hx + ex - 11 + fx, hy - 3)]),
                      "width": 2.5 * k, "paint": "black", "parent": "head", "shadow": 0})
    if hood:  # hood rim framing the face
        parts.append({"id": "hood_rim", "shape": "stroke",
                      "points": PL([(hx - head_r * 1.12, hy + head_r * 0.55), (hx - head_r * 1.1, hy - head_r * 0.5), (hx, hy - head_r * 1.18),
                                    (hx + head_r * 1.1, hy - head_r * 0.5), (hx + head_r * 1.12, hy + head_r * 0.55)]),
                      "width": 15 * k, "paint": hood, "parent": "head"})
    if hat == "nightcap":
        parts += [
            {"id": "hat", "shape": "poly", "smooth": True, "corners": [0, 4],
             "points": PL([(hx - head_r * 1.1, hy - head_r * 0.35), (hx - head_r * 0.7, hy - head_r * 1.25), (hx + head_r * 0.4, hy - head_r * 1.55),
                           (hx + head_r * 1.6, hy - head_r * 1.0), (hx + head_r * 1.1, hy - head_r * 0.35)]),
             "paint": hat_paint, "parent": "head"},
            {"id": "hat_band", "shape": "stroke", "points": PL([(hx - head_r * 1.15, hy - head_r * 0.38), (hx, hy - head_r * 0.48), (hx + head_r * 1.15, hy - head_r * 0.38)]),
             "width": 12 * k, "paint": "cream", "parent": "head"},
            {"id": "pompom", "shape": "circle", "c": P(hx + head_r * 1.65, hy - head_r * 0.92), "r": 11 * k, "paint": "red_cloth", "parent": "head"}]
    elif hat == "cap":
        parts += [
            {"id": "hat", "shape": "ellipse", "c": P(hx, hy - head_r * 0.62), "r": [head_r * 1.05 * k, head_r * 0.55 * k], "arc": [180, 360],
             "paint": hat_paint, "parent": "head"},
            {"id": "hat_brim", "shape": "ellipse", "c": P(hx + head_r * 0.6, hy - head_r * 0.6), "r": [head_r * 0.65 * k, 8 * k], "paint": hat_paint, "parent": "head"}]
    elif hat == "straw":
        parts += [
            {"id": "hat_brim", "shape": "ellipse", "c": P(hx, hy - head_r * 0.62), "r": [head_r * 1.6 * k, 12 * k], "paint": "wicker", "parent": "head"},
            {"id": "hat", "shape": "ellipse", "c": P(hx, hy - head_r * 0.7), "r": [head_r * 0.8 * k, head_r * 0.6 * k], "arc": [180, 360], "paint": "wicker", "parent": "head"}]
    legs = [{"part": "leg_l", "type": "swing", "deg": 16, "period": 0.62},
            {"part": "leg_r", "type": "swing", "deg": 16, "period": 0.62, "phase": 0.5}]
    arms = [{"part": "arm_l", "type": "swing", "deg": 12, "period": 0.62, "phase": 0.5},
            {"part": "arm_r", "type": "swing", "deg": 8, "period": 0.62}]
    blink = [{"part": ["eye_l", "eye_r"], "type": "blink", "period": 3.4}]
    rigs = {
        "idle": [{"part": "head", "type": "swing", "deg": 2.5, "period": 3.0},
                 {"part": ["arm_l", "arm_r"], "type": "swing", "deg": 2, "period": 3.0}] + blink,
        "walk": legs + arms + [{"part": "head", "type": "swing", "deg": 2, "period": 0.62}] + blink,
        "run": [dict(a, deg=a["deg"] * 2, period=0.36) for a in legs + arms] + blink,
        "talk": [{"part": "head", "type": "swing", "deg": 3, "period": 1.6},
                 {"part": "mouth_open", "type": "squash", "amp": 0.4, "period": 0.28},
                 {"part": "arm_l", "type": "swing", "deg": 10, "period": 1.6}] + blink,
        "wave": [{"part": "arm_l", "type": "rotate", "to": 150, "t": [0.0, 0.15]},
                 {"part": "arm_l", "type": "swing", "deg": 16, "period": 0.6}] + blink,
        "cheer": [{"part": "arm_l", "type": "rotate", "to": 150, "t": [0, 0.12]},
                  {"part": "arm_r", "type": "rotate", "to": -150, "t": [0, 0.12]},
                  {"part": ["arm_l", "arm_r"], "type": "swing", "deg": 10, "period": 0.5}] + blink,
        "scared": [{"part": "arm_l", "type": "rotate", "to": 125, "t": [0, 0.1]},
                   {"part": "arm_r", "type": "rotate", "to": -125, "t": [0, 0.1]},
                   {"part": "head", "type": "swing", "deg": 4, "period": 0.2}],
        "chop": [{"part": "arm_r", "type": "swing", "deg": 35, "period": 0.8}] + blink,
    }
    return {"parts": parts, "rigs": rigs}


TEMPLATES = {"person": person}


def expand_template(spec: dict) -> dict:
    """{"template": "person", "params": {...}, "extra_parts": [...], "rigs": {...}} -> actor spec."""
    if "template" not in spec:
        return spec
    base = TEMPLATES[spec["template"]](**spec.get("params", {}))
    parts = base["parts"]
    for extra in spec.get("extra_parts", []):
        before = extra.pop("before", None) if isinstance(extra, dict) else None
        if before:
            idx = next((i for i, p in enumerate(parts) if p.get("id") == before), len(parts))
            parts.insert(idx, extra)
        else:
            parts.append(extra)
    base["rigs"].update(spec.get("rigs", {}))
    return base
