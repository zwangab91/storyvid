# Storyboard spec (storyvid v1)

A storyboard is one JSON object. The renderer turns it into a narrated, painted-paper
collage video: bold hand-painted tissue-paper textures, cut-paper shapes with soft paper
shadows, mostly white page backgrounds, simple paper-puppet animation.

## Canvas and conventions
- Page is **1920 x 1080 design units**, x to the right, **y downward**. Angles in degrees, positive = clockwise.
- Captions cover roughly **y > 950**; keep faces and key action above that. Ground lines usually sit at y 760-880 and
  characters stand with their feet at y 860-930.
- Animation times are **fractions of the scene** (`"t": [0.2, 0.6]`) because scene length follows the narration.
  Periods (`"period"`) are in seconds.
- Draw order: items sorted by `z` (default 0), then list order. Later = on top.

## Top level
```json
{
  "title": "The Tortoise and the Hare",
  "subtitle": "optional second line on the title card (e.g. English title of a Chinese story)",
  "language": "en" | "zh",
  "voice": "af_heart",            // en: af_heart, af_bella, af_nicole, af_sarah, bf_emma, bf_isabella, bm_fable, bm_george, am_michael ...; zh: melo
  "speed": 0.92,                  // narration speed
  "credit": "Adapted from Aesop's Fables",   // small line on the end card
  "music": {"style": "musicbox" | "pluck", "seed": 4, "tempo": 84},   // or false for no music
  "paints": { "name": PAINT, ... },          // story-specific paints
  "actors": { "name": ACTOR, ... },          // story-specific characters and props
  "title_scene": {"background": BG, "items": [ITEM, ...]},   // art under the title letters (title sits at y~330)
  "scenes": [SCENE, ...],                    // 8-15 scenes is typical
  "end_scene": {"background": BG, "items": [ITEM, ...]},     // art under "The End" / "完"
  "title_card": true, "end_card": true                       // set false to skip the generated cards
}
```

## Scene
```json
{
  "narration": "One to three sentences read aloud and captioned.",
  "subtitle_en": "zh stories only: English translation with the SAME number of sentences.",
  "background": "paper" (default white page) | PAINT,
  "camera": {"zoom": [1.0, 1.06], "center": [[960, 540], [1000, 560]], "ease": "inout",
             "shake": {"amp": 6, "t": [0.5, 0.8]}},
  "hold": 0.5,                    // optional extra seconds after the narration
  "items": [ITEM, ...]
}
```
Default camera is a slow push-in from zoom 1.0 to 1.035. Close-ups: zoom 1.3-2.0 with `center` on the subject.

## Item (something placed in a scene)
```json
{"actor": "hare", "at": [560, 860], "scale": 0.9, "rot": 0, "flip": false, "z": 0, "seed": 0,
 "opacity": 1.0, "rig": "run" | ["run", "blink"] | [{"name": "run", "t": [0, 0.5]}],
 "anims": [ANIM, ...], "part_anims": [PART_ANIM, ...],
 "hide": ["part_id", ...], "reveal": ["hidden_part_id", ...]}
```
- `at` places the actor's origin. `scale` can be a number or `[sx, sy]`. `flip` mirrors horizontally (characters face right by default).
- Inline one-off shapes: `{"part": PART}` with absolute coordinates (no `at` needed), e.g. a path or a pond.
- `scatter` repeats an item: `{"actor": "grass_tuft", "rig": "sway", "scatter": {"n": 8, "area": [x0, y0, x1, y1], "scale": [0.7, 1.0], "seed": 2, "min_dist": 60}}`.
- `origin`: pivot point (actor-local) for the item's rotation and scale. Defaults to the actor origin.
- **A character appears once at a time.** A person, or a story actor with an `idle`, `walk`, `run`, `fly` or `talk` rig,
  may not be on screen twice at the same moment. Show repeated action with one item: a `path` with several loops, and
  `part_anims` `show` windows to change what it carries. Copies that take turns are fine (`show` windows, or one off
  screen while the other is on). Mark an actor `"character": false` if copies are meant (a flock).

## Animations (ANIM; for items, and for parts via part_anims/rigs)
Timed (use `t`): `move {to:[x,y]}`, `by {d:[dx,dy]}`, `path {points:[[x,y],...], orient?}`, `enter {from: "left"|"right"|"top"|"bottom"|[x,y]}`,
`exit {to: side|[x,y]}`, `scale {to}`, `rotate {to}`, `fade {to}`, `pop` (appear with overshoot), `grow` (appear, no overshoot),
`vanish`, `show {t:[a,b]}` (visible only in window), `flip {at}` (turn around at fraction).
Looping (optional `t` window, `period` seconds, `phase` 0..1): `hop {height}`, `bob {amp, dir:[dx,dy]}`, `sway {deg}`, `swing {deg}`,
`spin {speed deg/s}`, `pulse {amp}`, `squash {amp}`, `shake {amp}`, `drift {amp}` (clouds), `float_up {dist, dir}` (Zzz, bubbles, sweat).
Easing: `"ease": "linear" | "in" | "out" | "inout" | "sine" | "back" | "bounce"`.
Moves chain: a later `move` starts where the earlier one ended.
Part-only: `blink {period}` (squashes the part vertically for a moment).

## Actor
```json
{"parts": [PART, ...], "rigs": {"run": [PART_ANIM, ...], "idle": [...]}}
```
Parts are drawn in list order. Coordinates are actor-local design units with the origin at the feet (bottom centre)
for standing things, or the centre for sky things. Build characters at roughly real page size (a child ~330 tall, an
adult ~400, a rabbit ~360 incl. ears) and place them with `scale`.

PART_ANIM = ANIM + `"part": "id"` or `"part": ["id1", "id2"]`.

### Person template (paper-doll, front facing)
```json
"girl": {"template": "person", "params": {
   "height": 330, "child": true, "skin": "skin_light", "hair": "hair_brown", "hair_style": "short|long|bun|bob|bald",
   "top": "blue_cloth", "dress": true, "bottom": "blue_cloth", "apron": "white", "hood": "red_cloth", "shawl": null,
   "hat": "nightcap|cap|straw", "hat_paint": "white", "beard": "hair_brown", "glasses": false, "shoes": "black",
   "holding": "basket|axe|staff", "facing": 0.0},
 "extra_parts": [PART, ...], "rigs": {...}}
```
Rigs: idle, walk, run, talk (needs `"reveal": ["mouth_open"], "hide": ["mouth"]`), wave, cheer, scared, chop.
Part ids: leg_l/r, shoe_l/r, body, apron, arm_l/r, hand_l/r, neck, head, eye_l/r, cheek_l/r, mouth, mouth_open (hidden),
hair, hair_back, bun, beard, hat, cape, hood_rim, basket, axe_handle, staff ...

## Part (one cut-paper shape)
```json
{"id": "body", "shape": SHAPE..., "paint": PAINT, "pivot": [x, y], "parent": "id",
 "edge": "cut"|"torn"|"clean", "shadow": 1.0, "opacity": 1.0, "hidden": false,
 "tex_scale": 1.0, "tex_rot": deg, "cut": [SHAPE, ...]}
```
Shapes:
- `{"shape": "circle", "c": [x, y], "r": 40}`, `{"shape": "ellipse", "c": [x, y], "r": [rx, ry], "rot": deg, "arc": [a0, a1]?}`
- `{"shape": "rect", "c": [x, y], "size": [w, h], "radius": 8, "rot": deg}` or `"xywh": [x, y, w, h]`
- `{"shape": "poly", "points": [[x, y], ...], "smooth": true, "corners": [i, ...]}` smooth = curve through points; `corners` stay sharp
- `{"shape": "stroke", "points": [[x, y], ...], "width": 12 | [w_start, w_end], "cap": "round"|"flat"}` painted line, legs, stems, paths
- `{"shape": "blob", "c": [x, y], "r": [rx, ry], "jag": 0.18, "shape_seed": 3}` organic lump (bushes, rocks, tree crowns)
- `{"shape": "star", "c": [x, y], "r": [r_in, r_out], "n": 5, "smooth": false}`
- `{"shape": "cloud", "c": [x, y], "r": [rx, ry], "puffs": 5}`
- `{"shape": "wave", "x0": -200, "x1": 2120, "y": 700, "amp": 30, "wavelength": 300, "bottom": 1280}` band with a wavy top
- `{"shape": "text", "text": "z", "font": "title", "size": 60}`
- `{"shape": "image", "src": "layers/wing.png", "xywh": [x, y, w, h], "clip": [[x, y], ...]}` a piece of generated
  artwork used as a paper part. The image's alpha is the cut line; `xywh` places the whole image in actor units
  (default: 1 unit per pixel at 0,0); optional `clip` polygon and `cut` shapes trim it. `src` is relative to the
  storyboard file. Pivots, parents, rigs and shadows (default 0.6) work as for drawn parts.
`cut` subtracts shapes (crescent moon, holes, bites). `parent` makes a part follow another part's animation; `pivot`
is the rotation point for swings (shoulder, hip, neck, wing root).

## Paint
A preset name, a story paint name, a hex colour, or a recipe:
```json
{"colors": ["#base", "#streak1", "#streak2", "#accent"], "style": "streaks|swirl|dabs|sponge|cells|stripes|dots|speckle|flat",
 "angle": 30, "scale": 1.0, "contrast": 1.0, "period": 40, "cells": 60}
```
First colour = base, the rest are brushed on top. `cells` (shells, scales, cracked earth, patchwork: last colour = seams),
`stripes` (cloth, planks: `period` px), `dots` (polka, lace), `speckle` (splatter), `flat` (eyes, small details).
Presets: PRESET_LIST

## Built-in props (actors)
LIBRARY_LIST

## Style guide
- Think picture book spread: a few big, bold shapes on a white page; 1 main action per scene.
- Every character needs eyes (black dot + tiny white glint) and a simple readable silhouette built from 8-25 parts.
- Give characters paints with 4-5 related colours plus one surprising accent (e.g. purple streaks in brown fur).
- Animate everything a little (rig + one or two item anims); use camera zoom for emphasis, `shake` for drama.
- Keep stories gentle for children; violence happens off-page.
