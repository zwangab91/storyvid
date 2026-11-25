"""Side-by-side: Paperwing's playback of a keyframe (formula from src/utils/cinemaEngine.ts) vs the hybrid render."""
import math, subprocess, numpy as np, cv2, json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to hybrid/
from PIL import Image, ImageDraw, ImageFont
FPS, PW, PH = 24, 960, 540
cover = cv2.cvtColor(cv2.imread("src/jingwei_cover.jpg"), cv2.COLOR_BGR2RGB)
tl = json.load(open("out/hybrid_jingwei.timeline.json"))
total = tl[-1]["start"] + tl[-1]["dur"]
cut = tl[1]["start"] + 0.3                     # Paperwing plays scenes back to back
segs = [(0.0, cut, 1), (cut, total, 2)]         # (start, end, scene.index)

def paperwing_frame(t):
    s0, s1, idx = next(sg for sg in segs if sg[0] <= t < sg[1] or sg is segs[-1])
    p = min(1.0, max(0.0, (t - s0) / (s1 - s0)))
    k = PW / 1280.0                              # their canvas is 1280x720
    nh, nw = cover.shape[:2]
    cover_scale = max(1280 / nw, 720 / nh)
    stepped = math.floor(p * 48) / 48
    direction = -1 if idx % 2 == 0 else 1
    scale = cover_scale * (1.03 + 0.045 * p) * k
    ox = direction * (p - 0.5) * 32 * k
    oy = math.sin(stepped * math.pi * 2) * 5 * k
    rot = direction * math.sin(stepped * math.pi) * 0.0045
    c, s = math.cos(rot) * scale, math.sin(rot) * scale
    M = np.float32([[c, -s, PW / 2 + ox - (c * nw / 2 - s * nh / 2)], [s, c, PH / 2 + oy - (s * nw / 2 + c * nh / 2)]])
    img = cv2.warpAffine(cover, M, (PW, PH), flags=cv2.INTER_LINEAR, borderValue=(250, 248, 243))
    alpha = max(0.05, min(1, p / 0.08, (1 - p) / 0.08))
    paper = np.array([250, 248, 243], np.float32)
    return (img * alpha + paper * (1 - alpha)).astype(np.uint8)

font = ImageFont.truetype("../assets/fonts/Andika-Bold.ttf", 30)
small = ImageFont.truetype("../assets/fonts/Andika-Regular.ttf", 24)
canvas = Image.new("RGB", (1920, 1080), (244, 242, 236))
d = ImageDraw.Draw(canvas)
d.text((480, 205), "Paperwing playback", font=font, fill=(40, 36, 32), anchor="mm")
d.text((480, 240), "one keyframe per scene + Ken Burns drift", font=small, fill=(110, 104, 98), anchor="mm")
d.text((1440, 205), "Hybrid", font=font, fill=(40, 36, 32), anchor="mm")
d.text((1440, 240), "same art, cut into rigged paper pieces", font=small, fill=(110, 104, 98), anchor="mm")
base = np.asarray(canvas).copy()
dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", "out/hybrid_jingwei.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-",
                        "-i", "out/hybrid_jingwei.mp4", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", "out/compare_paperwing_vs_hybrid.mp4"], stdin=subprocess.PIPE)
n = 0
while True:
    raw = dec.stdout.read(1920 * 1080 * 3)
    if len(raw) < 1920 * 1080 * 3:
        break
    hyb = cv2.resize(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3), (PW, PH), interpolation=cv2.INTER_AREA)
    fr = base.copy()
    fr[270:270 + PH, 0:PW] = paperwing_frame(n / FPS)
    fr[270:270 + PH, 960:960 + PW] = hyb
    enc.stdin.write(fr.tobytes())
    n += 1
enc.stdin.close(); enc.wait()
print("frames", n)
