"""Step 2: turn the masks into RGBA paper pieces in plate coordinates (1376x768).

Prototype: the part outlines and pivots in RIG were picked by hand by looking at the plate
(the job a vision model would do in a full pipeline). Writes layers/*.png and layers_sheet.jpg.
"""
import cv2, numpy as np, json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to hybrid/
im = cv2.imread("src/jingwei_cover.jpg")
H, W = im.shape[:2]
M = np.load("masks.npy", allow_pickle=True).item()
bird, sea, top = M["bird"], M["sea"], M["top"]
paper_bgr = cv2.cvtColor(np.uint8([[M["paper"]]]), cv2.COLOR_LAB2BGR)[0, 0]

def poly_mask(pts):
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.array(pts, np.int32)], 1); return m

def save(name, rgb, alpha):
    a = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.8)
    out = np.dstack([rgb, np.clip(a * 255, 0, 255).astype(np.uint8)])
    cv2.imwrite(f"layers/{name}.png", out)

os.makedirs("layers", exist_ok=True)
# --- rig: polygons + pivots (plate px), chosen by looking at the plate, as a vision model would
RIG = {
  "wing_up":  {"poly": [(440,70),(575,95),(665,150),(716,210),(690,252),(640,280),(598,304),(545,312),(500,258),(470,196),(445,125)], "pivot": [662, 262]},
  "wing_low": {"poly": [(640,352),(700,318),(782,328),(810,362),(806,425),(796,525),(770,652),(712,654),(684,568),(644,488),(624,428),(624,384)], "pivot": [716, 348]},
  "tail":     {"poly": [(604,342),(594,420),(564,458),(500,482),(396,480),(390,402),(418,340),(480,326),(560,322)], "pivot": [596, 388]},
  "pebble":   {"poly": [(906,250),(922,250),(931,269),(947,281),(954,302),(946,330),(918,342),(892,334),(886,310),(894,286),(902,272)], "pivot": [915, 300]},
}
JOINT = 26  # px of overlap kept on the body side of each joint
body = bird.copy()
for k, r in RIG.items():
    pm = poly_mask(r["poly"])
    part = bird & pm
    save(k, im, part)
    if k == "pebble":
        body[pm > 0] = 0
        continue
    # keep a strip of the part on the body near the pivot so a swinging joint never opens a hole
    dist = np.hypot(*(np.mgrid[0:H, 0:W][::-1] - np.array(r["pivot"])[:, None, None]))
    body[(pm > 0) & (dist > JOINT * 2.2)] = 0
save("body", im, body)
# --- sea: fill where the bird hid it by cloning sea from further along the same rows
sea_rgb = im.copy()
hole = (bird > 0) & (sea > 0)
hsv_src = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
for dx in (330, -330, 600, -600):
    ys, xs = np.where(hole)
    src_x = xs + dx
    ok = (src_x >= 0) & (src_x < W)
    ok[ok] &= ~hole[ys[ok], src_x[ok]] & (sea[ys[ok], src_x[ok]] > 0) & (bird[ys[ok], src_x[ok]] == 0)
    hs = hsv_src[ys[ok], src_x[ok]]
    seaish = ((hs[:, 0] > 98) & (hs[:, 0] < 130)) | (hs[:, 1] < 60)
    idx = np.where(ok)[0]; ok[idx[~seaish]] = False
    sea_rgb[ys[ok], xs[ok]] = im[ys[ok], src_x[ok]]
    hole[ys[ok], xs[ok]] = False
sea_rgb = cv2.inpaint(sea_rgb, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
seam = cv2.dilate(((bird > 0) & (sea > 0)).astype(np.uint8), np.ones((3, 3))) - cv2.erode(((bird > 0) & (sea > 0)).astype(np.uint8), np.ones((3, 3)))
blur = cv2.GaussianBlur(sea_rgb, (0, 0), 1.5); sea_rgb[seam > 0] = blur[seam > 0]
save("sea", sea_rgb, sea)
# --- backdrop: paper + mountain; bird and sea removed; mountain extended down behind the sea line
mtn = M["mountain"].astype(np.uint8)
mtn = cv2.dilate(mtn, np.ones((3, 3), np.uint8))
rng = np.random.default_rng(3)
grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.8) * 2.2
tone = cv2.resize(cv2.GaussianBlur(rng.standard_normal((H // 16, W // 16)).astype(np.float32), (0, 0), 3), (W, H)) * 3.0
bg = np.clip(paper_bgr[None, None, :].astype(np.float32) + (grain + tone)[..., None], 0, 255).astype(np.uint8)
bg[mtn > 0] = im[mtn > 0]
yy = np.arange(H)[:, None]
band = ((sea > 0) & (yy < top[None, :] + 40) & (np.arange(W)[None, :] > 820) & (mtn == 0)).astype(np.uint8)
bg = cv2.inpaint(bg, band, 9, cv2.INPAINT_TELEA)
cv2.imwrite("layers/backdrop.png", bg)
json.dump({k: {"pivot": v["pivot"]} for k, v in RIG.items()}, open("layers/rig.json", "w"))
# preview: exploded view of the pieces
prev = np.full((H, W, 3), 255, np.uint8)
for k, off in (("body", (0, 0)), ("tail", (-40, 30)), ("wing_low", (30, 60)), ("wing_up", (-40, -45)), ("pebble", (60, 50))):
    rgba = cv2.imread(f"layers/{k}.png", cv2.IMREAD_UNCHANGED)
    Mt = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
    rgba = cv2.warpAffine(rgba, Mt, (W, H))
    a = rgba[..., 3:4] / 255.0
    prev = (prev * (1 - a) + rgba[..., :3] * a).astype(np.uint8)
seaa = cv2.imread("layers/sea.png", cv2.IMREAD_UNCHANGED)
sheet = np.vstack([np.hstack([cv2.resize(bg, (688, 384)), cv2.resize(seaa[..., :3] * (seaa[..., 3:4] / 255.0) + 255 * (1 - seaa[..., 3:4] / 255.0), (688, 384)).astype(np.uint8)]),
                   np.hstack([cv2.resize(prev, (688, 384)), cv2.resize(im, (688, 384))])])
cv2.imwrite("layers_sheet.jpg", sheet)
print("ok")
