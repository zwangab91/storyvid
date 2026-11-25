"""Download the narration models into ./models (no Hugging Face access needed).

English: Kokoro-82M v1.0 (q8 ONNX, Apache-2.0) + voice styles, from the npm packages
         `kokoro-q8-shards` and `kokoro-local-runtime`.
Mandarin: MeloTTS zh (sherpa-onnx conversion, MIT) from the npm package `@3sln/donki-lle-voice-zh`.

    python scripts/fetch_models.py
"""
import base64
import glob
import hashlib
import io
import json
import os
import re
import tarfile
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")
REG = "https://registry.npmjs.org"


def tarball(pkg, version):
    meta = json.load(urllib.request.urlopen(f"{REG}/{pkg.replace('/', '%2F')}/{version}"))
    print(f"downloading {pkg}@{meta['version']} ...")
    data = urllib.request.urlopen(meta["dist"]["tarball"]).read()
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def kokoro():
    d = os.path.join(M, "kokoro")
    os.makedirs(d, exist_ok=True)
    onnx = os.path.join(d, "kokoro-v1.0-q8.onnx")
    if not os.path.exists(onnx):
        t = tarball("kokoro-q8-shards", "1.0.0")
        parts = sorted((m for m in t.getmembers() if m.name.endswith(".bin")), key=lambda m: m.name)
        blob = b"".join(t.extractfile(m).read() for m in parts)
        assert hashlib.sha256(blob).hexdigest().startswith("fbae9257e1e05ffc"), "kokoro checksum mismatch"
        open(onnx, "wb").write(blob)
    if not glob.glob(os.path.join(d, "*.bin")):
        t = tarball("kokoro-local-runtime", "0.1.0")
        for m in t.getmembers():
            if m.name.startswith("package/voices/") and m.name.endswith(".bin"):
                open(os.path.join(d, os.path.basename(m.name)), "wb").write(t.extractfile(m).read())
    voices = {os.path.basename(f)[:-4]: np.fromfile(f, dtype=np.float32).reshape(-1, 1, 256)
              for f in glob.glob(os.path.join(d, "*.bin"))}
    np.savez(os.path.join(d, "voices.npz"), **voices)
    print(f"kokoro ready ({len(voices)} voices)")


def melo_zh():
    d = os.path.join(M, "melo_zh")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "model.onnx")):
        print("melo_zh ready")
        return
    t = tarball("@3sln/donki-lle-voice-zh", "0.1.1")
    files = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    voice = files["package/voice.js"].decode("utf-8")
    meta = json.loads(voice[voice.index("{"): voice.rindex("}") + 1])
    for name, info in meta["files"].items():
        data = b""
        for p in info["parts"]:
            js = files["package/c/" + p.split("/c/")[1]].decode("utf-8")
            data += base64.b64decode(re.match(r'\s*export default "([A-Za-z0-9+/=]*)";?\s*$', js).group(1))
        assert len(data) == info["size"], name
        open(os.path.join(d, name), "wb").write(data)
    json.dump(meta["textMap"], open(os.path.join(d, "textmap.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("melo_zh ready")


if __name__ == "__main__":
    kokoro()
    melo_zh()
