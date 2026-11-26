"""Narration (Kokoro for English, MeloTTS for Mandarin), music bed synthesis, and mixing."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter, resample_poly

SR = 44100
MODELS = os.environ.get("STORYVID_MODELS",
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))


# --------------------------------------------------------------------------- text splitting

def split_sentences(text: str, lang: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if lang.startswith("zh"):
        parts = re.findall(r"[^。！？!?；]+[。！？!?；]*[”」』]?", text)
    else:
        parts = re.findall(r".+?(?:[.!?…][\"”’]?(?=\s+[A-Z\"“‘(])|$)", text)
    return [p.strip() for p in parts if p.strip()]


def split_caption(sentence: str, lang: str, max_len: int) -> list[str]:
    """Break a long sentence into caption-sized chunks at commas / clause breaks."""
    if len(sentence) <= max_len:
        return [sentence]
    seps = r"(?<=[，、,;；：:—])" if lang.startswith("zh") else r"(?<=[,;:—])\s+"
    pieces = [p for p in re.split(seps, sentence) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        joiner = "" if lang.startswith("zh") else " "
        cand = (cur + joiner + p).strip() if cur else p.strip()
        if len(cand) > max_len and cur:
            chunks.append(cur)
            cur = p.strip()
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:  # still too long: hard split on words / characters
        while len(c) > max_len * 1.35:
            if lang.startswith("zh"):
                out.append(c[:max_len])
                c = c[max_len:]
            else:
                cut = c.rfind(" ", 0, max_len)
                cut = cut if cut > 0 else max_len
                out.append(c[:cut])
                c = c[cut:].strip()
        out.append(c)
    return out


# --------------------------------------------------------------------------- TTS

class Narrator:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._kokoro = None
        self._melo = None
        self._textmap = None

    def _kokoro_engine(self):
        if self._kokoro is None:
            import glob
            from kokoro_onnx import Kokoro
            vz = os.path.join(MODELS, "kokoro", "voices.npz")
            if not os.path.exists(vz):
                voices = {os.path.basename(f)[:-4]: np.fromfile(f, dtype=np.float32).reshape(-1, 1, 256)
                          for f in glob.glob(os.path.join(MODELS, "kokoro", "*.bin"))}
                np.savez(vz, **voices)
            self._kokoro = Kokoro(os.path.join(MODELS, "kokoro", "kokoro-v1.0-q8.onnx"), vz)
        return self._kokoro

    def _melo_engine(self):
        if self._melo is None:
            import sherpa_onnx
            d = os.path.join(MODELS, "melo_zh")
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=f"{d}/model.onnx", lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt"),
                    num_threads=2),
                rule_fsts=",".join(f"{d}/{x}.fst" for x in ["phone", "date", "number"]),
                max_num_sentences=1)
            self._melo = sherpa_onnx.OfflineTts(cfg)
            self._textmap = json.load(open(f"{d}/textmap.json", encoding="utf-8"))
        return self._melo

    def synth_one(self, text: str, lang: str, voice: str, speed: float) -> np.ndarray:
        key = hashlib.sha1(json.dumps([text, lang, voice, speed, "v2"]).encode()).hexdigest()[:16]
        path = os.path.join(self.cache_dir, key + ".wav")
        if os.path.exists(path):
            a, _ = sf.read(path, dtype="float32")
            return a
        if lang.startswith("zh"):
            eng = self._melo_engine()
            t = "".join(self._textmap.get(ch, ch) for ch in text)
            # the Mandarin voice only knows , . ! ? … as pauses: map the rest, drop quote marks
            for a, b in (("：", "，"), ("；", "，"), ("—", "，"), ("、", "，")):
                t = t.replace(a, b)
            t = "".join(ch for ch in t if ch not in "“”「」『』‘’\"《》")
            au = eng.generate(t, sid=0, speed=speed)
            a = np.asarray(au.samples, np.float32)
            if au.sample_rate != SR:
                a = resample_poly(a, SR, au.sample_rate).astype(np.float32)
        else:
            eng = self._kokoro_engine()
            kl = "en-gb" if voice.startswith("b") else "en-us"
            a, sr = eng.create(text, voice=voice, speed=speed, lang=kl)
            g = math.gcd(SR, sr)
            a = resample_poly(a, SR // g, sr // g).astype(np.float32)
        a = _trim(a)
        sf.write(path, a, SR)
        return a

    def narrate(self, text: str, lang: str, voice: str, speed: float, gap: float = 0.32):
        """-> (audio, [(start_s, end_s, sentence), ...])"""
        sents = split_sentences(text, lang)
        out, spans, t = [], [], 0.0
        for i, s in enumerate(sents):
            a = self.synth_one(s, lang, voice, speed)
            a = _level(a)
            out.append(a)
            spans.append((t, t + len(a) / SR, s))
            t += len(a) / SR
            if i < len(sents) - 1:
                g = gap + (0.12 if s.endswith(("。", ".", "!", "！", "?", "？")) else 0)
                out.append(np.zeros(int(g * SR), np.float32))
                t += g
        return (np.concatenate(out) if out else np.zeros(1, np.float32)), spans


# --------------------------------------------------------------------------- pre-generated narration (Gemini TTS)

def read_tts_wav(path: str) -> np.ndarray:
    """A generated narration clip as mono float32 at SR, without the provenance block Google appends to it.

    Gemini 3.8 TTS WAVs end with about 3 KB of C2PA content credentials that the RIFF header counts as samples, so a
    plain player sounds a short burst of loud noise at the end. Speech ends in silence, and read as 16-bit samples the
    block is mostly text: loud and lopsided (positive), unlike speech. So if what follows the last 60 ms of silence
    is short, loud and lopsided, cut it."""
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    fr = max(1, int(0.005 * sr))
    n = len(a) // fr
    if n > 20:
        seg = a[: n * fr].reshape(n, fr)
        quiet = np.sqrt((seg ** 2).mean(1)) < 0.01
        lo, i, cut = max(0, n - int(0.8 * sr / fr)), n - 1, None
        while i >= lo:  # the last stretch of >= 60 ms of silence near the end
            if quiet[i]:
                j = i
                while j >= lo and quiet[j]:
                    j -= 1
                if i - j >= 12:
                    cut = i + 1
                    break
                i = j
            else:
                i -= 1
        if cut is not None:
            tail = a[cut * fr:]
            r = float(np.sqrt((tail ** 2).mean())) if len(tail) else 0.0
            # what follows is short, loud and lopsided (text bytes are positive samples): the appended block
            if len(tail) <= 0.4 * sr and r > 0.1 and float(tail.mean()) > 0.15 * r:
                a = a[: cut * fr]
    if sr != SR:
        g = math.gcd(SR, sr)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a.astype(np.float32)


def _weight(s: str, lang: str) -> int:
    return max(1, len(re.findall(r"[㐀-鿿]", s)) if lang.startswith("zh") else len(re.findall(r"\w", s)))


def align_sentences(a: np.ndarray, sents: list[str], lang: str, min_pause: float = 0.1) -> list[tuple]:
    """Sentence spans in a clip that reads several sentences in one go, from its pauses.

    The expected boundaries come from each sentence's share of the characters; each is snapped to a pause, preferring
    long pauses (sentence ends are usually the longest) near the expected time, in order. -> [(start_s, end_s, s)]"""
    dur = len(a) / SR
    if len(sents) <= 1:
        return [(0.0, dur, sents[0] if sents else "")]
    fr = int(0.01 * SR)
    n = len(a) // fr
    rms = np.sqrt((a[: n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9
    quiet = 20 * np.log10(rms / rms.max()) < -38
    pauses, k = [], 0
    while k < n:
        if quiet[k]:
            j = k
            while j < n and quiet[j]:
                j += 1
            if (j - k) * 0.01 >= min_pause and k > 0 and j < n:
                pauses.append((k * 0.01, j * 0.01))
            k = j
        else:
            k += 1
    w = np.array([_weight(s, lang) for s in sents], float)
    expect = list(np.cumsum(w)[:-1] / w.sum() * dur)
    need = len(sents) - 1
    if len(pauses) < need:
        cuts = [(t, t) for t in expect]
    else:
        # dynamic programme: pick `need` pauses in order, cost = distance to expected time - bonus for length
        P = len(pauses)
        cost = [[abs((p0 + p1) / 2 - e) - 0.6 * min(p1 - p0, 1.0) for (p0, p1) in pauses] for e in expect]
        INF = 1e18
        best = [[INF] * P for _ in range(need)]
        back = [[-1] * P for _ in range(need)]
        for j in range(P):
            best[0][j] = cost[0][j]
        for i in range(1, need):
            run_min, run_arg = INF, -1
            for j in range(P):
                if j - 1 >= 0 and best[i - 1][j - 1] < run_min:
                    run_min, run_arg = best[i - 1][j - 1], j - 1
                if run_arg >= 0:
                    best[i][j] = run_min + cost[i][j]
                    back[i][j] = run_arg
        j = min(range(P), key=lambda x: best[need - 1][x])
        picks = []
        for i in range(need - 1, -1, -1):
            picks.append(j)
            j = back[i][j]
        cuts = [pauses[p] for p in reversed(picks)]
    spans, t = [], 0.0
    for s, (c0, c1) in zip(sents, cuts + [(dur, dur)]):
        spans.append((t, c0, s))
        t = c1
    return spans


class ClipNarrator:
    """Narration from clips made by storyvid.soundtrack: one clip per scene, looked up by the scene's text."""

    def __init__(self, folder: str):
        with open(os.path.join(folder, "assets.json"), encoding="utf-8") as f:
            rows = json.load(f).get("assets", [])
        self.folder = folder
        self.by_text = {r["text"].strip(): os.path.join(folder, r["file"]) for r in rows if r.get("type") == "speech"}
        self.music = next((os.path.join(folder, r["file"]) for r in rows if r.get("type") == "music"), None)
        self.raw_rms: list[float] = []  # voiced level of each clip before levelling (sets the music balance)

    def missing(self, texts: list[str]) -> list[str]:
        return [t for t in texts if t.strip() and t.strip() not in self.by_text]

    def narrate(self, text: str, lang: str, *_):
        path = self.by_text.get(text.strip())
        if not path:
            raise KeyError(f"no narration clip for: {text[:40]}... (re-run storyvid.soundtrack plan and generate)")
        a = _trim(read_tts_wav(path))
        self.raw_rms.append(_voiced_rms(a))
        a = _level(a)
        return a, align_sentences(a, split_sentences(text, lang), lang)


def _voiced_rms(a: np.ndarray) -> float:
    fr = int(0.03 * SR)
    n = len(a) // fr
    if n < 1:
        return float(np.sqrt((a ** 2).mean()) + 1e-9)
    rms = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(1))
    v = rms[rms > rms.max() * 0.15]
    return float(np.sqrt((v ** 2).mean())) if len(v) else 1e-4


def _trim(a, thr=0.004):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    s, e = max(0, idx[0] - int(0.03 * SR)), min(len(a), idx[-1] + int(0.08 * SR))
    return a[s:e]


def _level(a, target_rms=0.085):
    fr = int(0.03 * SR)
    n = len(a) // fr
    if n < 1:
        return a
    rms = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(1))
    voiced = rms[rms > rms.max() * 0.15]
    g = target_rms / max(1e-4, float(np.sqrt((voiced ** 2).mean())) if len(voiced) else 1e-4)
    a = a * min(g, 8.0)
    peak = np.abs(a).max()
    if peak > 0.95:
        a = a * (0.95 / peak)
    return a.astype(np.float32)


# --------------------------------------------------------------------------- music bed

def _note_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _music_box(f, dur, vel):
    t = np.arange(int(dur * SR)) / SR
    env = lambda tau: np.exp(-t / tau)
    a = (np.sin(2 * np.pi * f * t) * env(0.9) + 0.22 * np.sin(2 * np.pi * f * 2.756 * t) * env(0.25)
         + 0.07 * np.sin(2 * np.pi * f * 5.404 * t) * env(0.08))
    a[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return (a * vel).astype(np.float32)


def _pluck(f, dur, vel, rng):
    """Karplus-Strong plucked string (zither-like)."""
    N = max(2, int(SR / f))
    n = int(dur * SR)
    x = np.zeros(n, np.float32)
    burst = rng.uniform(-1, 1, N).astype(np.float32)
    burst = np.convolve(burst, [0.5, 0.5], "same")
    x[:N] = burst
    g = 0.996
    a = np.zeros(N + 2)
    a[0] = 1
    a[N] -= g * 0.5
    a[N + 1] -= g * 0.5
    y = lfilter([1.0], a, x).astype(np.float32)
    # gentle pitch bend-in on attack, characteristic of zither ornaments, via short fade
    y[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    return y * vel * 0.6


def _pad(freqs, dur):
    t = np.arange(int(dur * SR)) / SR
    a = np.zeros_like(t)
    for f in freqs:
        for det in (-0.6, 0.6):
            a += np.sin(2 * np.pi * (f + det) * t)
    att = np.minimum(1, t / 1.2)
    rel = np.minimum(1, (dur - t) / 1.2)
    return (a * att * np.clip(rel, 0, 1) / (2 * len(freqs))).astype(np.float32)


def synth_music(duration: float, seed: int = 0, style: str = "musicbox", tempo: float = 84.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int((duration + 2) * SR)
    out = np.zeros(n, np.float32)
    beat = 60.0 / tempo
    root = 60 + int(rng.integers(-3, 4))  # around middle C
    penta = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    prog = [[0, 4, 7], [9, 12, 16], [5, 9, 12], [7, 11, 14]] if style == "musicbox" else \
        [[0, 7, 12], [-3, 4, 9], [2, 9, 14], [-5, 2, 7]]
    # a 2-bar motif, re-used with variations
    motif = []
    deg = int(rng.integers(2, 6))
    for i in range(8):
        deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 1, 1, 2]), 0, len(penta) - 1))
        motif.append(deg)
    bar = 4 * beat
    t, bar_i = 1.0, 0
    while t < duration:
        chord = prog[bar_i % 4]
        pad = _pad([_note_freq(root - 12 + c) for c in chord], bar + 1.2) * 0.10
        s = int(t * SR)
        out[s:s + len(pad)] += pad[: max(0, n - s)]
        phrase = motif if (bar_i // 4) % 2 == 0 else [int(np.clip(d + rng.choice([-1, 0, 1]), 0, 9)) for d in motif]
        half = phrase[(bar_i % 2) * 4:(bar_i % 2) * 4 + 4]
        for k, d in enumerate(half):
            if rng.random() < 0.18 and k > 0:
                continue  # breathe
            nt = t + k * beat + (beat / 2 if (k == 3 and rng.random() < 0.3) else 0)
            f = _note_freq(root + 12 + penta[d])
            vel = 0.20 * rng.uniform(0.8, 1.0)
            tone = _music_box(f, 2.6, vel) if style == "musicbox" else _pluck(f, 2.6, vel * 1.3, rng)
            s = int(nt * SR)
            if s < n:
                out[s:s + len(tone)] += tone[: n - s]
        t += bar
        bar_i += 1
    # small room reverb
    ir_t = np.arange(int(1.4 * SR)) / SR
    ir = rng.standard_normal(len(ir_t)).astype(np.float32) * np.exp(-ir_t / 0.32).astype(np.float32)
    ir /= np.sqrt((ir ** 2).sum())
    wet = fftconvolve(out, ir)[:n].astype(np.float32)
    mix = 0.75 * out + 0.45 * wet
    # fades
    fi, fo = int(1.0 * SR), int(3.0 * SR)
    end = int(duration * SR)
    mix = mix[:end]
    mix[:fi] *= np.linspace(0, 1, fi)
    mix[-fo:] *= np.linspace(1, 0, fo)
    return mix / (np.abs(mix).max() + 1e-6) * 0.5


# --------------------------------------------------------------------------- generated music (Lyria)

def load_music(path: str) -> np.ndarray:
    """Any audio file -> stereo float32 (n, 2) at SR, decoded with ffmpeg."""
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def loop_music(clip: np.ndarray, duration: float, xfade: float = 2.5, fade_in: float = 1.0,
               fade_out: float = 3.0) -> np.ndarray:
    """Repeat a clip to `duration` seconds with equal-power crossfades at the joins, fading in and out at the ends."""
    clip = clip[: len(clip) - int(0.05 * SR)] if len(clip) > 10 * SR else clip  # drop encoder padding
    n, x = int(duration * SR), int(xfade * SR)
    out = np.zeros((n, clip.shape[1]), np.float32)
    up = np.sin(np.linspace(0, np.pi / 2, x))[:, None].astype(np.float32)
    down = np.cos(np.linspace(0, np.pi / 2, x))[:, None].astype(np.float32)
    pos, first = 0, True
    while pos < n:
        c = clip.copy()
        if not first:
            c[:x] *= up
        c[-x:] *= down
        end = min(n, pos + len(c))
        out[pos:end] += c[: end - pos]
        pos += len(c) - x
        first = False
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    out[:fi] *= np.linspace(0, 1, fi)[:, None]
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return out


def mix_bed(narration: np.ndarray, music: np.ndarray, voiced: list[tuple[float, float]], gain_free: float,
            gain_voice: float, ramp: float = 0.6) -> np.ndarray:
    """Stereo mix: narration in the centre, music at `gain_free`, lowered to `gain_voice` through every voiced
    stretch (whole scenes, as Paperwing does, so the music doesn't pump in the pauses). -> (n, 2) float32"""
    n = len(narration)
    m = np.zeros((n, 2), np.float32)
    m[: min(n, len(music))] = music[:n]
    target = np.full(n, gain_free, np.float32)
    for a, b in voiced:
        target[max(0, int(a * SR)):min(n, int(b * SR))] = gain_voice
    k = max(1, int(ramp * SR))
    c = np.cumsum(np.concatenate([[0.0], target]), dtype=np.float64)
    smooth = np.empty(n, np.float32)
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k // 2 + 1, 0, n)
    smooth[:] = (c[hi] - c[lo]) / (hi - lo)
    out = narration[:, None] + m * smooth[:, None]
    peak = np.abs(out).max()
    return (out / peak * 0.95 if peak > 0.95 else out).astype(np.float32)


def mix_tracks(narration: np.ndarray, music: np.ndarray | None, music_db: float = -21.0, duck_db: float = -7.0):
    n = len(narration)
    if music is None:
        return np.clip(narration, -1, 1)
    m = np.zeros(n, np.float32)
    m[: min(n, len(music))] = music[:n]
    # sidechain ducking from narration envelope
    fr = int(0.02 * SR)
    k = n // fr + 1
    pad = np.zeros(k * fr, np.float32)
    pad[:n] = np.abs(narration)
    env = (pad.reshape(k, fr).max(1) > 0.02).astype(np.float32)
    # smooth: fast attack, slow release
    sm = np.zeros_like(env)
    v = 0.0
    for i, e in enumerate(env):
        v = max(e, v * 0.93)
        sm[i] = v
    sm = np.repeat(sm, fr)[:n]
    gain = 10 ** (music_db / 20) * (1 - sm * (1 - 10 ** (duck_db / 20)))
    out = narration + m * gain * 2.0
    peak = np.abs(out).max()
    return (out / peak * 0.95 if peak > 0.95 else out).astype(np.float32)
