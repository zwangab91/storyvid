# storyvid: architecture and tech stack

storyvid turns a story (English or Mandarin) into a narrated picture-book video in a hand-painted tissue-paper
collage look. An LLM plans the film as data; local code renders every frame, voices the narration and scores it
(or, optionally, the narration and music come from Gemini TTS and Lyria, generated once per story).
This document covers the stack, how data moves through it, and why it is built this way. Usage is in the
[README](../README.md); the storyboard format is in [STORYBOARD_SPEC.md](STORYBOARD_SPEC.md).

This document describes the pipeline as of the commit it is in. Any commit that changes the stack updates it in
the same commit, so `git log -p docs/ARCHITECTURE.md` shows how the architecture evolved.

---

## 1. Overview

**What it produces:** a 1080p24 H.264 MP4 with AAC audio, an `.srt` subtitle file and a contact sheet. The four pilot
storyboards in `stories/` render to:

| Pilot | Language | Scenes | Story-specific actors | Voice | Length |
|---|---|---|---|---|---|
| The Tortoise and the Hare | en | 12 | 5 | Kokoro `bf_emma` | 1:51 |
| Little Red Riding Hood | en | 15 | 13 | Kokoro `af_heart` | 2:26 |
| 精卫填海 Jingwei Fills the Sea | zh + English subtitles | 12 | 8 | MeloTTS | 1:55 |
| 夸父逐日 Kuafu Chases the Sun | zh + English subtitles | 13 | 8 | MeloTTS | 1:49 |

**Design goals**

- **Continuous, coherent motion:** every frame is rendered from a scene graph. Characters keep the same cut-paper
  pieces across scenes, and nothing is a looped clip or a re-shown still.
- **Editable intermediate:** the storyboard JSON is the single source of truth. A person or a model can edit it and
  re-render without touching code.
- **Local and repeatable:** after the storyboard is written, rendering, speech and music run offline on CPU. The same
  JSON and seeds give the same video.
- **Bilingual:** native-language narration, with sentence-aligned English subtitles for Chinese stories.

---

## 2. Tech stack at a glance

| Layer | Technology | Code |
|---|---|---|
| Language and runtime | Python 3.10+, CLI entry points (`python -m storyvid.pipeline`, `python -m storyvid.storyboard`) | `storyvid/` |
| Planning (story to storyboard) | Claude, through the `claude -p` CLI (Claude subscription) or the Anthropic API (`anthropic` SDK, default `claude-sonnet-5-5`) | `storyboard.py` |
| Storyboard format | JSON scene graph: paints, actors made of parts, rigs, scenes, items, animations, camera | `docs/STORYBOARD_SPEC.md` |
| Paper textures | NumPy noise fields (fBm, anisotropic streaks, warps) shaped by colour recipes; OpenCV resizing and blurs | `paper.py` |
| Shapes and cut edges | Polygon primitives (ellipse, rect, star, blob, cloud, stroke, wave band, Catmull-Rom splines) and edge roughening | `geometry.py` |
| Rasterising and compositing | OpenCV `warpAffine` on premultiplied float32 RGBA sprites with mip chains; Pillow for title and caption text | `render.py` |
| Animation | Easing curves; item and part animations timed as fractions of the scene; rigs; camera zoom, pan and shake | `anim.py`, `render.py` |
| Reusable props | Scenery, sky, plants, water, and a `person` paper-doll template | `library.py` |
| English speech | Kokoro-82M v1.0, 8-bit quantised ONNX, via `kokoro-onnx` (US and UK voices) | `audio.py` |
| Mandarin speech | MeloTTS zh (VITS) via `sherpa-onnx` | `audio.py` |
| Music | Procedural synthesis in NumPy/SciPy: music box or Karplus-Strong plucked zither over a pentatonic motif, pads and a synthetic room reverb | `audio.py` |
| Generated narration (optional) | Gemini 3.8 Flash-Lite TTS over REST, voice Kore with a written storyteller direction, one request per scene; captions timed from the pauses | `soundtrack.py`, `audio.py` |
| Generated music (optional) | Lyria 3 Clip (30 s, 44.1 kHz stereo MP3) through the Gemini Interactions API; looped at a searched loop point with crossfades | `soundtrack.py`, `audio.py` |
| Audio processing | SciPy (`resample_poly`, `fftconvolve`), `soundfile`; side-chain ducking (procedural bed) or per-scene ducking (generated music); stereo mix; ffmpeg `loudnorm` (-16 LUFS, -1.5 dBTP) | `audio.py`, `pipeline.py` |
| Video encoding | ffmpeg: libx264 (`veryfast`, CRF 17, yuv420p) per segment, concat demuxer, AAC 192 kbps, `+faststart` | `pipeline.py` |
| Parallelism | `concurrent.futures.ProcessPoolExecutor`, one worker and one ffmpeg pipe per CPU core | `pipeline.py` |
| Caching | On-disk `.cache/` (textures as float16 `.npy`, TTS per sentence as WAV), in-memory sprite cache per worker | `paper.py`, `audio.py`, `render.py` |
| Model download | `scripts/fetch_models.py`: about 170 MB of voice models from npm-registry packages, checked by size and checksum | `scripts/` |
| Fonts | Fredoka (titles) and Andika (captions), SIL OFL, bundled; system Noto CJK / PingFang / Songti for Chinese | `assets/fonts/` |
| Hybrid prototype | Image-model art (from Paperwing Studio) cut into rigged `"shape": "image"` parts; OpenCV GrabCut and keying | `hybrid/` |
| Asset generation | Gemini image API (`gemini-nano-banana-2.1`) over REST with `requests`, run in GitHub Actions: asset manifests, reference images for consistency, content-keyed cache, per-run image cap, spend log; audio manifests (narration, music) go through the same workflow | `assets.py`, `soundtrack.py`, `.github/workflows/assets.yml` |
| Cut-out keying | OpenCV: board colour from the image border, CIE Lab distance with a soft ramp, hole filling, connected pieces to RGBA crops plus a cuttability report | `cutout.py` |
| Rigging and actor swap | NumPy/OpenCV: joints found as the straight cut across a limb's long axis, limbs placed joint-to-attach-point, figures fitted to the replaced actor's footprint; rigs adopted by part name; a cast file maps a storyboard's actors to generated art | `rig.py`, `hybrid/cast/` |

External requirements: `ffmpeg` on `PATH`. Storyboard generation also needs the `claude` CLI or `ANTHROPIC_API_KEY`;
rendering an existing storyboard does not.

---

## 3. System diagram

```text
                       ┌───────────────────────────── Stage 1: plan (LLM) ─────────────────────────────┐
 story text ─────────► │ prompt = spec + example storyboard + story ──► Claude ──► storyboard draft    │
                       │        ▲                                                         │            │
                       │        └── errors / contact sheet ◄── validate: build + render ◄─┘            │
                       └───────────────────────────────────────────────────────────────┬───────────────┘
                                                                                       │ storyboard.json
                       ┌──────────────── Stage 2: timeline + audio (local) ────────────▼───────────────┐
                       │ sentences ─► Kokoro / MeloTTS ─► scene durations, caption timings             │
                       │ procedural music bed ─► ducked under narration ─► mix.wav                     │
                       └───────────────────────────────────────────────────────────────┬───────────────┘
                                                                                       │ timed scenes
                       ┌──────────────── Stage 3: render (local, CPU, N processes) ────▼───────────────┐
                       │ paints ─► textures   parts ─► roughened polygons ─► textured, shadowed sprites│
                       │ per frame: camera ─► items ─► rigs / parents ─► blit onto paper ─► captions   │
                       │ worker k renders frames [f_k, f_k+1) straight into its own x264 segment       │
                       └───────────────────────────────────────────────────────────────┬───────────────┘
                                                                                       │ segments + mix.wav
                       ┌──────────────── Stage 4: assemble (ffmpeg) ───────────────────▼───────────────┐
                       │ concat segments ─► mux audio + loudnorm ─► story.mp4, story.srt, contact sheet│
                       └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline stages in detail

### Stage 1: storyboard (`storyboard.py`)

- The prompt contains the full storyboard spec, one complete pilot storyboard as an example, and the story. Claude
  returns a scene list (10-14 scenes by default) with narration (plus `subtitle_en` for Chinese), story-specific
  paints, actors built from cut-paper parts, rigs, animations and camera moves.
- **Validation renders the draft.** Every scene is built and rendered at 384×216 at three moments. Exceptions (an
  unknown paint, a missing rig or part, bad geometry), narration/subtitle sentence-count mismatches and more copies
  of a character on screen than the story has go back to
  Claude for up to 3 repair rounds.
- **Never more of a character than the story has.** Every story actor declares who it depicts (`"character"`: an id
  shared by pose variants, or `false` for scenery and props) and how many the story has (`"count"`: 1 by default, a
  number such as the three pigs, or `"many"` for a herd or flock). An undeclared actor is a validation error, so
  identity and number are decided when the storyboard is written rather than guessed. Each item that depicts a
  character is then rendered on its own at 48 moments, on black and on white; the difference is its coverage, with
  camera, paths and hidden parts included. A copy counts while its own animation state has it at least 60% opaque and
  it shows on at least 20 pixels of the 384×216 frame; any moment with more copies than the count is a problem (for an
  individual, two). A dissolve passes (the two never both reach 60%), copies that take turns pass, and an item marked
  `"duplicate"` (a reflection) is left out. Cases are in `tests/test_duplicate_characters.py`. Left as is, because
  they fail safe (a repair round, never a doubled character): a copy completely hidden behind scenery still counts.
  A double shorter than one sample (about 2% of a scene) can slip through.
- **Optional visual critique** (`--critique N`; the `make` command defaults to 1): the contact sheet image is shown to
  Claude, which returns a revised storyboard. A revision is kept only if it still validates.
- Backends: `claude-cli` (the `claude` CLI in print mode) is picked when installed; otherwise the Anthropic API.

### Stage 2: timeline and audio (`audio.py`, `pipeline.py`)

- Narration is split into sentences and each sentence is synthesised separately at 44.1 kHz, cached by a hash of
  text, language, voice and speed. Sentence boundaries give exact caption timing.
- **The narration drives timing.** Scene duration = 0.75 s lead-in + narration + optional hold + 0.95 s tail, with a
  minimum of 3.2 s (4.5 s for the title card, 4.0 s for the end card). Consecutive scenes overlap by a 0.6 s
  cross-dissolve.
- Captions: each sentence is cut into chunks (at most 64 characters in English, 24 in Chinese) whose screen time is
  proportional to their length. For Chinese stories the matching English sentence stays on screen under every
  chunk of its sentence. Consecutive captions hard-cut; a run of captions fades in and out.
- Music: a procedural bed (music box for Western tales, plucked zither for Chinese ones) whose level is set by
  `music.db` (default -21), ducked by a further 7 dB while the voice is speaking (fast attack, slow release). The final mix is loudness-normalised by
  ffmpeg to -16 LUFS integrated, -1.5 dBTP.
- **Generated soundtrack (optional).** When the storyboard names a `soundtrack` folder (made by `soundtrack.py`, see
  section 7), each scene is voiced by its Gemini clip instead: one clip reads the whole scene, so the model plans the
  phrasing across sentences. Sentence spans for the captions come from the pauses: each boundary expected from the
  sentences' share of characters is snapped, in order, to a nearby pause, preferring long ones (a dynamic programme);
  caption chunks within a long sentence are placed on its comma pauses the same way.
  Generated music is a 30-second clip that usually fades out at its end, so it is left at the point before the fade
  whose last two seconds sound most like its first two (spectral similarity and level), and looped with equal-power
  crossfades. It is mixed in stereo at Paperwing Studio's balance: the voice at full level, the music at 0.28 of its
  own level, times 0.45 through every narrated scene (whole scenes, so it does not pump in the pauses), scaled by the
  gain that levelling applied to the voice.

### Stage 3: collage renderer (`paper.py`, `geometry.py`, `render.py`, `anim.py`)

- **Design space:** 1920×1080 units, y down. Output resolution scales from it (`--height 720` for drafts).
- **Paints to textures:** each paint is a recipe of a few colours and a brush technique (streaks, swirl, dabs,
  cells, speckle, stripes, dots, sponge, flat). Textures are float RGB squares (1024², or 2048² for large sprites),
  deterministic per recipe and seed, and cached in memory and on disk.
- **Parts to sprites:** a part's shape becomes a polygon, its edge is roughened by cut style (`clean`, `cut`,
  `torn`), the texture is mapped inside it, and a soft shadow is baked in (strength 0.2, offset (4, 5), blur 3.5).
  The result is a premultiplied RGBA sprite rasterised at a scale quantised to steps of 2^(1/4), with a mip chain
  for strong zoom-outs. `"shape": "image"` parts load an RGBA image instead of painting a texture.
- **Frames:** for each frame the renderer computes the camera matrix (eased zoom and centre, optional shake), then
  for each item its transform from its animations. Animation times are fractions of the scene, so a storyboard
  stays valid at any narration length. Each part gets a local transform about its pivot from the item's rigs and
  part animations and is chained through its parent parts. Sprites are composited onto the paper background with
  one `warpAffine` per part.
- **Transitions and text:** cross-dissolves use a smoothstep blend of the two scenes' frames; captions and titles
  are Pillow-rendered sprites composited the same way.

### Stage 4: assembly (`pipeline.py`)

- The frame range is split into one contiguous block per worker (CPU count by default). Each worker builds its own
  renderer and streams raw RGB frames into its own ffmpeg/x264 process, so no frames touch disk.
- If a worker dies (for example out of memory), the remaining segments are re-rendered one at a time.
- Segments are joined with the concat demuxer (no re-encode), then the audio is muxed and loudness-normalised.
  Outputs: `<name>.mp4`, `<name>.srt`, `<name>.contact.jpg`.

---

## 5. Data model

- **Storyboard JSON** (see the [spec](STORYBOARD_SPEC.md)): `title`, `language`, `voice`, `speed`, `music`, `paints`,
  `actors`, `scenes`, optional `title_scene` / `end_scene` art, and `title_card` / `end_card` switches.
- **Actor:** a named list of parts, each with an optional `id`, `pivot` and `parent`, plus named `rigs` (lists of
  part animations such as a walk cycle or a wing flap). Story-specific actors live in the storyboard; reusable props
  live in `library.py`.
- **Scene (runtime):** the storyboard scene plus derived fields: `_start`, `_dur` and `_nar_start` (timeline),
  `_spans` (sentence timings), `_captions`, and `_items` (expanded items).
- **Sprite:** premultiplied float32 RGBA, its origin offset, its raster scale and lazily built mips. Sprites are
  keyed by (actor, part, raster scale, seed).

---

## 6. Key design decisions and trade-offs

### Decision 1: the LLM writes a scene graph; code paints the pixels
- **Chosen:** Claude outputs a storyboard of cut-paper parts, rigs and animation; a deterministic renderer draws
  every frame.
- **Why:** continuity comes by construction. The same sprites represent a character in every scene, motion is
  continuous at 24 fps, and the result is editable JSON at near-zero cost per frame.
- **Trade-off:** the look is limited to what the procedural paints and shapes can express. It is flatter and simpler
  than image-model art. The hybrid direction (Section 7) addresses this.

### Decision 2: narration-driven timing with relative animation times
- **Chosen:** scene length comes from the synthesised voice; animation and camera times are fractions of a scene.
- **Why:** changing voice, speed or wording never breaks choreography, and captions line up with speech exactly.

### Decision 3: validate by rendering, not just by schema
- **Chosen:** each storyboard draft is built and rendered at low resolution, and errors are returned to the model;
  optionally the model also critiques a contact sheet.
- **Why:** most LLM mistakes (a rig referencing a missing part, an unknown paint, broken geometry) only surface when
  the scene is built. Feeding real errors back fixes them in a round or two.

### Decision 4: local, offline rendering and speech on CPU
- **Chosen:** ONNX TTS models, procedural music and a NumPy/OpenCV renderer, with no GPU and no API calls after the
  storyboard.
- **Why:** runs on a laptop, costs nothing per render, and is reproducible. Sprite caching keeps a 1080p frame at
  roughly 0.05-0.1 s on one core, and frames are rendered in parallel across cores.
- **Trade-off:** voice quality and music are good but not studio-grade, and the Mandarin voice supports only a small
  set of pause punctuation (others are mapped to commas).

### Decision 5: parallel segments piped straight into x264
- **Chosen:** contiguous frame blocks per process, each piping raw frames into its own encoder, then a lossless
  concat.
- **Why:** no intermediate image files, near-linear speed-up with cores, and memory bounded by evicting sprites of
  scenes that have finished (about two scenes resident per worker).

### Relation to Paperwing Studio
Paperwing Studio (a sibling project) generates one painted keyframe per beat with an image model and plays it with
a 2D cutout drift or a short AI video clip. Its frames are richer, but holding one still per scene makes motion feel
repetitive. storyvid's frames are simpler, but its motion is continuous. The hybrid direction combines the two.

---

## 7. Hybrid direction: generate the pieces, not the frames

An image model paints each character and location once, as separate pieces; storyvid cuts them into rigged paper
parts and animates every frame on the narration timeline.

```text
 story ─► [storyboard + asset list: Claude] ─► [asset generation: image model, once per character / layer]   (built)
                                                        │
                                                        ▼
                     [cut + rig: keying, joints, attach points from the planner, joint check]             (built)
                                                        │
                                                        ▼
                     storyboard with "shape": "image" parts ─► Stages 2-4 unchanged                         (ready)
```

| Piece | Status |
|---|---|
| Image parts in the renderer (pivots, parents, rigs, clip/cut masks, baked shadow, camera, caching) | Ready |
| Cutting a finished illustration into parts (`hybrid/extract_layers.py`, `make_layers.py`) | Prototype, hand-tuned to one image |
| Asset-generation stage (`storyvid/assets.py`, the "Generate assets" workflow) | Built; used for the 19 images of 精卫 |
| Generated narration and music (`storyvid/soundtrack.py`, same workflow) | Built; used for 精卫 (13 narration clips and one music clip, about $0.07) |
| Cutting pieces off a plain board (`storyvid/cutout.py`) | Built; tested on synthetic sheets made from real generated pieces |
| Rigging and actor swap (`storyvid/rig.py`, cast files) | Built; joints automatic, attach points from the cast file |
| Storyboard step that writes the asset list and cast file | Not built: for 精卫 both were written by the planner by hand |

First full film: 精卫填海 (115 s, 14 scenes) rendered from `stories/jingwei.json` unchanged, with all 19 actors and
the three backdrops swapped for generated art: 19 images for about $1.14 in total. With the generated soundtrack
(Gemini narration in Paperwing's voice and a Lyria score, about $0.07) the same storyboard runs 3:07, since the
storyteller reads at about 2.4 characters a second against MeloTTS's 4.2.

**Asset stage.** A manifest lists assets (a character, its parts sheet, a pose, a scenery strip), each a prompt plus
optional reference images, usually an earlier asset so the character keeps its look. An asset is identified by a hash
of model, size, aspect, prompt, variant and references; re-running skips assets that already exist, so only new or
edited ones are paid for, and editing an asset regenerates the assets that reference it. A run refuses to start above
a set number of new images, `--dry-run` prints the plan and estimated cost, and `assets.json` logs the tokens and
approximate cost of every call. A reference can be an image from another manifest's run (`@run/file.png`), and an
asset can carry its own style text (backdrops are full-bleed rather than on a board).

**Why GitHub Actions.** The sandbox this pipeline is developed in can reach only package registries and GitHub, not
image APIs. The "Generate assets" workflow (manual trigger, dry run by default) runs on GitHub's runners with the
repository secret `GEMINI_API_KEY`, cuts every image with `cutout.py`, and commits the results to the
`generated-assets` branch so `main` stays small. Standard runners are free for public repositories.

**Rigging and the actor swap.** A storyboard names actors (`jingwei`, `sea`, `sun`, ...) and refers to their parts
and rigs by name. `rig.py` builds a generated-art actor with the same interface, so the storyboard itself does not
change: figures are assembled from a parts sheet (each limb's joint is found automatically as the straight cut that
runs across its long axis at one end; the limb is placed so that joint sits on an attach point of the root piece),
props come from one piece or a keyed region of a props sheet, and strips (sea, ground) are hung at the replaced
actor's layer heights. Each is scaled to the replaced actor's footprint and origin. Its rigs are copied by part name,
entries for parts it lacks are dropped and listed, and the cast file can alias names or replace rig values (painted
wings flap ±20° rather than the procedural ±38°). Scene backgrounds can be swapped for generated backdrops.
`rig.preview` renders each actor at four moments of its rig as the joint check. The cast file holds what the planner
decides by looking at the cut overlays: which piece is which part, attach points, pivots for pieces without a straight
cut, draw order and tuned rig values.

**Soundtrack.** `python -m storyvid.soundtrack plan <storyboard>` writes an audio manifest: one narration request per
voiced scene (title included) with the model, voice and direction Paperwing Studio uses (Gemini 3.8 Flash-Lite TTS,
Kore, "Warm, gentle, expressive ... storyteller with unhurried pacing and poetic wonder"), and one Lyria 3 Clip request
from the storyboard's `music.prompt` plus an instrumental-only suffix. The "Generate assets" workflow runs it like an
image manifest (cached by request, capped, cost-logged; narration is billed at about 25 audio tokens a second, $6 per
million through 2026 and $12 from January 2027; Lyria 3 Clip is $0.04 a clip). A cast file's `soundtrack` names the run
folder, and `rig.py` copies it next to the hybrid storyboard. Gemini's WAVs carry a C2PA provenance record in their own
RIFF chunk; copies that count it as audio (AI Studio's log viewer makes such copies) play it as a burst of noise, so
`audio.read_tts_wav` cuts it when present.

**Cut-out.** Pieces are painted on plain cream board. The board colour and noise are measured on the image border;
pixels far enough from it in Lab are paper; enclosed holes are filled because paper is opaque. Each connected piece
becomes an RGBA crop, and the report flags anything that would make cutting a guess: a non-plain board, a piece cut
off by the image edge, or board-coloured paint just inside an outline.

Details and the proof of concept: [hybrid/README.md](../hybrid/README.md).

---

## 8. Repository layout

```text
storyvid/
├── storyvid/
│   ├── storyboard.py   # story text -> storyboard JSON with Claude; validate, repair, critique
│   ├── pipeline.py     # CLI (render, make); scenes, timeline, captions, frame maker, parallel encode, mux
│   ├── render.py       # sprites, rasterising (drawn and image parts), actors and rigs, camera, compositing
│   ├── paper.py        # procedural painted-paper textures, paint presets, texture cache
│   ├── geometry.py     # shape primitives -> polygons, splines, cut-edge roughening
│   ├── anim.py         # easing and animation evaluation (scene-fraction timing)
│   ├── library.py      # built-in props and the person template
│   ├── audio.py        # Kokoro / MeloTTS narration, generated clips, music synthesis and loops, ducking and mixing
│   ├── soundtrack.py   # audio manifests: Gemini TTS narration per scene and Lyria music, generated in Actions
│   ├── assets.py       # asset manifests -> generated images (Gemini), cache, cost cap and log
│   ├── cutout.py       # cut pieces off a plain board -> RGBA crops and a cuttability report
│   └── rig.py          # joints, figure assembly, prop/strip actors, rig adoption, storyboard swap, joint check
├── stories/            # four pilot storyboards
├── docs/               # this document and the storyboard spec
├── hybrid/             # generated-art prototype; asset manifests (hybrid/assets/) and cast files (hybrid/cast/)
├── .github/workflows/  # assets.yml: the "Generate assets" workflow
├── scripts/fetch_models.py
├── assets/fonts/       # Fredoka, Andika (SIL OFL)
└── requirements.txt
```

Runtime directories (git-ignored): `models/` (voice models), `.cache/` (textures, TTS), `out/` (renders),
`generated/` (local asset runs). Assets generated in CI live on the `generated-assets` branch.

---

## 9. Configuration

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Use the Anthropic API backend for storyboards |
| `GEMINI_API_KEY` | Image API key for asset generation: a repository secret for the workflow, or an environment variable for local runs |
| `STORYVID_MODELS` | Voice model directory (default `models/`) |
| `STORYVID_CACHE` | Texture and TTS cache directory (default `.cache/`) |
| `STORYVID_FONT_ZH`, `STORYVID_FONT_ZH_TITLE` | Override the Chinese caption and title fonts |

Style dials in code: `PRESETS` (paints) in `paper.py`, `EDGE_AMP` (cut roughness) and `SHADOW` in `render.py`,
`XFADE`, `LEAD` and `TAIL` (timing) in `pipeline.py`.
