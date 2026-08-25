#!/usr/bin/env python3
"""Montage: cadwork-IFC-Vordach (anthrazit) in Hausfoto, Kamera an Fluchtlinien gefittet."""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy import ndimage
import re

IFC = 'vordach.ifc'
PHOTO = 'haus_neu.jpg'
CAM = np.load('camera_D40.npy')        # f, D, yaw, pitch, yc, zc  (Fit an Wand-Homographie)

# ---------------- mini IFC parser (tessellierte Geometrie) ----------------
ents = {}
for line in open(IFC, encoding='utf-8', errors='replace'):
    m = re.match(r'#(\d+)=\s*IFC(\w+)\((.*)\);\s*$', line.strip())
    if m:
        ents[int(m.group(1))] = (m.group(2), m.group(3))

def parse_args(s):
    out, i, n = [], 0, len(s)
    def value():
        nonlocal i
        while i < n and s[i] in ' ,': i += 1
        if i >= n: return None
        c = s[i]
        if c == '(':
            i += 1
            lst = []
            while True:
                while i < n and s[i] in ' ,': i += 1
                if s[i] == ')':
                    i += 1
                    return lst
                lst.append(value())
        if c == "'":
            j = i + 1
            while s[j] != "'": j += 1
            v = s[i+1:j]; i = j + 1
            return ('str', v)
        if c == '#':
            j = i + 1
            while j < n and s[j].isdigit(): j += 1
            v = int(s[i+1:j]); i = j
            return ('ref', v)
        if c == '.':
            j = s.index('.', i+1)
            v = s[i:j+1]; i = j + 1
            return ('enum', v)
        if c in '$*':
            i += 1
            return None
        j = i
        while j < n and s[j] not in ',()': j += 1
        v = s[i:j].strip(); i = j
        try: return float(v)
        except ValueError: return ('tok', v)
    while i < n:
        while i < n and s[i] in ' ,': i += 1
        if i >= n: break
        out.append(value())
    return out

ARGS = {k: parse_args(argstr) for k, (typ, argstr) in ents.items()}

def ref(v): return v[1] if isinstance(v, tuple) and v[0] == 'ref' else None

def vec(v, default):
    if v is None: return np.array(default, float)
    return np.array(ARGS[ref(v)][0], float)

def axis_mat(pid):
    a = ARGS[pid]
    O = vec(a[0], [0., 0., 0.])
    Z = vec(a[1], [0., 0., 1.])
    X = vec(a[2] if len(a) > 2 else None, [1., 0., 0.])
    Z /= np.linalg.norm(Z)
    X = X - Z * np.dot(X, Z); X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = X, Y, Z, O
    return M

def placement_mat(pid):
    if pid is None: return np.eye(4)
    a = ARGS[pid]
    parent = placement_mat(ref(a[0])) if a[0] is not None else np.eye(4)
    return parent @ axis_mat(ref(a[1]))

elements = []
for eid, (typ, _) in ents.items():
    if typ not in ('BEAM', 'SLAB'): continue
    a = ARGS[eid]
    name = a[2][1] if isinstance(a[2], tuple) else ''
    M = placement_mat(ref(a[5]))
    rep = ARGS[ref(a[6])]
    polys = []
    for sr in rep[2]:
        sra = ARGS[ref(sr)]
        for item in sra[3]:
            fa = ARGS[ref(item)]
            pts = [np.array(p, float) for p in ARGS[ref(fa[0])][0]]
            gpts = [(M @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in pts]
            for f_ in fa[2]:
                idx = ARGS[ref(f_)][0]
                polys.append([gpts[int(i) - 1] for i in idx])
    elements.append((name, eid, polys))

# ---------------- Kamera (aus Fit) ----------------
F_, D_, TH, PH, YC, ZC = CAM
ct, st = np.cos(TH), np.sin(TH)
cp, sp_ = np.cos(PH), np.sin(PH)
FWD = np.array([-ct*cp, st*cp, sp_])
RGT = np.array([st, ct, 0.0])
UP = np.cross(RGT, FWD)
CPOS = np.array([D_, YC, ZC])
SS = 3

def proj(p):
    P = np.array([p[0], p[1], p[2]]) - CPOS
    den = P @ FWD
    u = 750 + F_ * (P @ RGT) / den
    v = 1000 - F_ * (P @ UP) / den
    return (u * SS, v * SS, den)

def proj1(p):
    u, v, _ = proj(p)
    return u / SS, v / SS

# ---------------- Gelaende: Pfostenfuesse loesen ----------------
DROP = -0.30                            # Vordach steht ~30 cm unter Tuerschwellen-Niveau
def t_wall(u):  return 1258 - 0.095 * (u - 256)
def t_front(u): return 1310 - 0.080 * (u - 300)

POSTS = {(0.06, 0.61): t_wall, (0.06, 6.19): t_wall,
         (3.94, 0.61): t_front, (3.94, 6.19): t_front}
GROUND = {}
for (px, py), tf in POSTS.items():
    g = -0.4
    for _ in range(6):
        u, v = proj1((px, py, g))
        g -= (tf(u) - v) / 130.0        # dv/dg ist negativ (~130 px/m)
    GROUND[(px, py)] = g
print("Gelaende je Pfosten:", {k: round(v, 2) for k, v in GROUND.items()})

def post_key(polys):
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    return (0.06 if min(xs) < 1.0 else 3.94, 0.61 if np.mean(ys) < 3.4 else 6.19)

faces = []
for name, eid, polys in elements:
    if name == 'Bodenplatte':
        continue
    g = GROUND.get(post_key(polys)) if name == 'Stiel' else None
    for poly in polys:
        pts = []
        for p in poly:
            x, y, z = p
            z = z + DROP
            if name == 'Stiel' and p[2] < 0.01:
                z = g
            pts.append((x, y, z))
        faces.append((pts, name))

# ---------------- Shading + Painter (anthrazit RAL 7016) ----------------
ANTHRAZIT = (64, 70, 76)
L = np.array([-0.30, 0.52, 0.80]); L /= np.linalg.norm(L)

def newell(pts):
    nrm = np.zeros(3)
    for i in range(len(pts)):
        a, b = np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
        nrm += np.cross(a, b)
    ln = np.linalg.norm(nrm)
    return nrm / ln if ln > 0 else nrm

img = Image.new('RGBA', (1500 * SS, 2000 * SS), (0, 0, 0, 0))
drw = ImageDraw.Draw(img)
rend = []
for pts, name in faces:
    n = newell(pts)
    cen = np.mean(pts, axis=0)
    if np.dot(n, CPOS - cen) < 0: n = -n
    pp = [proj(p) for p in pts]
    depth = sum(q[2] for q in pp) / len(pp)
    lam = max(0.0, float(np.dot(n, L)))
    b = 0.42 + 0.85 * lam ** 0.9
    col = tuple(int(min(255, c * b)) for c in ANTHRAZIT)
    edge = tuple(int(min(255, c * 1.28 + 7)) for c in col)
    rend.append((depth, [(q[0], q[1]) for q in pp], col + (255,), edge + (255,)))
rend.sort(key=lambda r: -r[0])
for _, pp, col, edge in rend:
    drw.polygon(pp, fill=col, outline=edge, width=2)
img = img.resize((1500, 2000), Image.LANCZOS)
img.save('render_ifc.png')

# ---------------- Grade + Composite ----------------
sp = np.array(img).astype(np.float32)
rgb, al = sp[..., :3], sp[..., 3] / 255.0

# hinterer rechter Pfosten verschwindet hinter dem Anhaenger
uBR, vBR = proj1((0.06, 6.19, 0.0))
x0, x1 = int(uBR) - 16, int(uBR) + 16
al[1156:, max(0, x0):x1] = 0.0

rgb *= np.array([0.96, 0.965, 0.98])
rng = np.random.default_rng(11)
rgb += rng.normal(0, 1.3, rgb.shape)
sp = np.dstack([np.clip(rgb, 0, 255), al[..., None] * 255]).astype(np.uint8)
sprite = Image.fromarray(sp, 'RGBA').filter(ImageFilter.GaussianBlur(0.35))

photo = Image.open(PHOTO).convert('RGB')
base = np.array(photo).astype(np.float32)

def draw_mask(shapes, blur):
    m = Image.new('L', photo.size, 0)
    d = ImageDraw.Draw(m)
    for kind, coords, val in shapes:
        if kind == 'poly': d.polygon(coords, fill=val)
        else: d.ellipse(coords, fill=val)
    return ndimage.gaussian_filter(np.array(m).astype(np.float32) / 255.0, blur)

bL = proj1((0.06, 0.0, 2.24 + DROP + 0.3))
bR = proj1((0.06, 6.8, 2.24 + DROP + 0.3))
band = []
steps = 12
for i in range(steps + 1):
    t = i / steps
    band.append(('poly', [(bL[0], bL[1] + 8 + 150 * t), (bR[0], bR[1] + 8 + 150 * t),
                          (bR[0], bR[1] + 8 + 150 * (t + 1 / steps)),
                          (bL[0], bL[1] + 8 + 150 * (t + 1 / steps))],
                 int(190 * (1 - t))))
wall = draw_mask(band, 22) * 0.15
ground = draw_mask([('ellipse', (280, 1170, 1340, 1330), 255)], 48) * 0.10

feet = {k: proj1((k[0], k[1], g)) for k, g in GROUND.items()}
print("Fuesse:", {k: (round(u), round(v)) for k, (u, v) in feet.items()})
contact = draw_mask(
    [('ellipse', (u - 40, v - 10, u + 40, v + 10), 255)
     for k, (u, v) in feet.items() if k != (0.06, 6.19)], 9) * 0.26
shade = np.clip(wall + ground + contact, 0, 0.4)
base *= (1 - shade[..., None])
out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
out.paste(sprite, (0, 0), sprite)
out.save('final_ifc.jpg', quality=93)
print("final_ifc.jpg gespeichert")
