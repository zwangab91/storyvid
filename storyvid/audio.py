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
