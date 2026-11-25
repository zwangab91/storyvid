# storyvid: story to painted-paper collage video

Turns a story into a narrated children's picture-book video in a hand-painted tissue-paper collage look: bold
brushy textures, scissor-cut shapes with soft paper shadows, white pages, paper-puppet animation, captions and a
gentle music bed. English and Mandarin.

```
story text ──► [1 storyboard: Claude] ──► storyboard.json ──► [2 narration: Kokoro / MeloTTS] ──► timeline
                     ▲   validate + repair (+ optional visual critique of a contact sheet)          │
                     └──────────────────────────────────────────────────────────────────────────────┤
                                     [3 collage renderer] ──► frames ──► [4 ffmpeg] ──► MP4 + SRT + contact sheet
```

1. **Storyboard** (`storyvid/storyboard.py`): Claude adapts the story into scenes, narration (plus English subtitles
   for Chinese stories), character designs built from cut-paper parts, props, animation and camera moves, all as JSON
   ([spec](docs/STORYBOARD_SPEC.md)). Every draft is rendered at low resolution; errors (unknown paint, missing rig,
   subtitle/narration sentence mismatch, ...) go back to the model for repair, and `--critique N` shows the model the
   contact sheet for visual fixes.
2. **Narration** (`storyvid/audio.py`): sentence-by-sentence TTS gives exact caption timing. English uses Kokoro-82M
   (28 voices), Mandarin uses MeloTTS. A procedural music bed (music box, or a plucked zither for Chinese tales) is
   ducked under the voice and the mix is loudness-normalised to -16 LUFS.
3. **Renderer** (`storyvid/paper.py`, `geometry.py`, `render.py`, `anim.py`): procedural painted papers (streaks,
   swirls, dabs, cells, stripes, dots, speckle), shapes with roughened cut edges and baked shadows, hierarchical
   actors with rigs, an animation engine, camera moves and shake, cross-dissolves and captions. Sprites are cached,
   so a 1080p frame takes about 0.05-0.1 s on one core.
4. **Assembly** (`storyvid/pipeline.py`): renders frame ranges in parallel straight into x264, then concatenates and
   muxes the audio. Also writes `.srt` subtitles and a contact sheet.

Scene timing follows the narration. Animation times are fractions of each scene, so a storyboard stays valid
whatever voice or speed you choose.

Tech stack, data flow and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup
```bash
pip install -r requirements.txt          # Python 3.10+
brew install ffmpeg                      # or apt install ffmpeg
python scripts/fetch_models.py           # ~170 MB of voice models, pulled from the npm registry
```
For storyboard generation you need either the Claude Code CLI (`claude`, uses your Claude subscription) or
`ANTHROPIC_API_KEY` plus `pip install anthropic`. Rendering an existing storyboard needs neither.

Chinese fonts: Noto CJK on Linux, PingFang/Songti on macOS, picked automatically. Override them with
`STORYVID_FONT_ZH` and `STORYVID_FONT_ZH_TITLE`.

## Use
```bash
# any story -> video (storyboard by Claude, then render)
python -m storyvid.pipeline make my_story.txt --title "The Lion and the Mouse" --lang en --out out/
python -m storyvid.pipeline make 后羿射日.txt --title 后羿射日 --subtitle "Hou Yi Shoots the Suns" --lang zh --out out/

# step by step
python -m storyvid.storyboard my_story.txt --title "..." --lang en -o stories/my_story.json --critique 1
python -m storyvid.pipeline render stories/my_story.json --out out/ --preview          # contact sheet in ~10 s
python -m storyvid.pipeline render stories/my_story.json --out out/ --stills 0.3,0.8 --only 2,5
python -m storyvid.pipeline render stories/my_story.json --out out/                    # full 1080p24 MP4
```
Options: `--height 720` for faster drafts, `--fps 12` for an on-twos cut-out feel, `--workers N`.

## Pilot storyboards (`stories/`)
| file | language | notes |
|---|---|---|
| `tortoise_and_hare.json` | en, voice bf_emma | Aesop; hare, tortoise and fox are custom paper rigs |
| `little_red_riding_hood.json` | en, voice af_heart | Grimm, gentle ending (Grandmother hides in the wardrobe; the woodcutter chases the wolf off) |
| `jingwei.json` | zh + English subtitles | 精卫填海, after 《山海经·北山经》: patterned head, white beak, red feet |
| `kuafu.json` | zh + English subtitles | 夸父逐日, after 《山海经·海外北经》: snake earrings, staff becomes the peach grove 邓林 |

## Hybrid prototype: generated art, procedural motion

`hybrid/` is a proof of concept for combining image-model artwork with this pipeline's motion. An image model paints
the pieces once (characters, set layers); storyvid cuts them into rigged paper parts and animates every frame on the
narration timeline, so nothing loops or repeats. The renderer side is general: any image can be a paper part
(`"shape": "image"`, see the [spec](docs/STORYBOARD_SPEC.md)). The cutting and rigging in `hybrid/` is hand-tuned to
one generated picture. The asset stage that paints pieces on plain board (`storyvid/assets.py`) and the cut-out that
separates them (`storyvid/cutout.py`) are built; they run in GitHub Actions ("Generate assets" in the Actions tab, dry
run first) with a `GEMINI_API_KEY` repository secret. See [hybrid/README.md](hybrid/README.md).

## Extending
- New paints: add a recipe to `PRESETS` in `paper.py`, or define one per story under `"paints"`.
- New reusable props: `LIBRARY` in `library.py` (they are listed to the model automatically).
- New character templates: add a function to `TEMPLATES` in `library.py` (see `person`).
- Different look: textures, edge roughness (`EDGE_AMP`) and shadow (`SHADOW`) are the main style dials.

## Credits and licences
Kokoro-82M (Apache-2.0), MeloTTS (MIT) via sherpa-onnx, Fredoka and Andika fonts (SIL OFL). The pilot stories are
retellings of public-domain tales. The style is generic painted-paper collage; the pipeline doesn't reproduce any
specific book's characters or artwork.
