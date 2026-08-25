#!/usr/bin/env python3
"""Montage: cadwork-IFC-Vordach in Hausfoto rendern (kalibrierte Kamera)."""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy import ndimage
import re, sys

IFC = '/root/.claude/uploads/9c543c24-f8f8-5118-b9fe-a3a73a31618c/e9293948-vordach.ifc'
PHOTO = 'attachments/haus_neu.jpg'

# ---------------- mini IFC parser (tessellated geometry) ----------------
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
    return np.array(ARGS[ref(v)][0], float)   # IFCCARTESIANPOINT/IFCDIRECTION -> coord list

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
    a = ARGS[pid]                      # IFCLOCALPLACEMENT(rel, axis)
    parent = placement_mat(ref(a[0])) if a[0] is not None else np.eye(4)
    return parent @ axis_mat(ref(a[1]))

elements = []                          # (name, [poly(list of 3d pts)])
for eid, (typ, _) in ents.items():
    if typ not in ('BEAM', 'SLAB'): continue
    a = ARGS[eid]
    name = a[2][1] if isinstance(a[2], tuple) else ''
    M = placement_mat(ref(a[5]))
    rep = ARGS[ref(a[6])]              # productdefinitionshape
    polys = []
    for sr in rep[2]:
        sra = ARGS[ref(sr)]            # shaperepresentation
        for item in sra[3]:
            fa = ARGS[ref(item)]       # polygonalfaceset(coords, closed, faces)
            pts = [np.array(p, float) for p in ARGS[ref(fa[0])][0]]
            gpts = [(M @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in pts]
            for f in fa[2]:
                idx = ARGS[ref(f)][0]
                polys.append([gpts[int(i) - 1] for i in idx])
    elements.append((name, eid, polys))

print("Elemente:", [(n, len(p)) for n, _, p in elements])

# ---------------- scene setup ----------------
DROP = -0.30                            # Gelaende liegt ~30 cm unter Tuerschwelle
GROUND = {(0.06, 0.61): -0.47, (0.06, 6.19): 0.20,
          (3.94, 0.61): -0.55, (3.94, 6.19): -0.13}

def post_key(polys):
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    x0 = 0.06 if min(xs) < 1.0 else 3.94
    y0 = 0.61 if np.mean(ys) < 3.4 else 6.19
    return (x0, y0)

faces = []                              # (pts3d, name)
for name, eid, polys in elements:
    if name == 'Bodenplatte':
        continue                        # Gelaende im Foto belassen
    g = GROUND.get(post_key(polys)) if name == 'Stiel' else None
    for poly in polys:
        pts = []
        for p in poly:
            x, y, z = p
            z = z + DROP
            if name == 'Stiel' and p[2] < 0.01:
                z = g                   # Pfostenfuss aufs lokale Gelaende
            pts.append((x, y, z))
        faces.append((pts, name))

# ---------------- camera (calibrated on haus_neu.jpg) ----------------
D = 22.0
SCALE_WALL = 128.7
F = SCALE_WALL * D
CAMW, CAMH = 0.47, 1.20                 # lateral / eye height (Tuerschwellen-Datum)
CX, VH = 750.0, 1010.0
SHEAR = 0.0416
SS = 3

def proj(p):
    x, y, z = p                         # x=Tiefe(0 Wand), y=Breite, z=Hoehe
    w = y - 3.4
    zd = D - x
    u = CX + F * (w - CAMW) / zd
    v = VH + F * (CAMH - z) / zd + SHEAR * (u - 252.0)
    return (u * SS, v * SS, zd)

# ---------------- shading + painter ----------------
COLORS = {'Stiel': (224, 216, 186), 'Kopfband': (224, 216, 186),
          'Sparren': (228, 220, 190), 'Pfette': (207, 198, 174)}
L = np.array([-0.30, 0.52, 0.80]); L /= np.linalg.norm(L)   # (x_depth, y_width, z_up)

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
CAMPOS = np.array([D, 3.4 + CAMW, CAMH])   # in ifc coords (x depth from wall... camera at x=D)
for pts, name in faces:
    n = newell(pts)
    cen = np.mean(pts, axis=0)
    view = CAMPOS - cen
    if np.dot(n, view) < 0: n = -n
    pp = [proj(p) for p in pts]
    depth = sum(q[2] for q in pp) / len(pp)
    lam = max(0.0, float(np.dot(n, L)))
    b = 0.40 + 0.72 * lam ** 0.85
    base = COLORS.get(name, (210, 205, 180))
    col = tuple(int(min(255, c * b)) for c in base)
    edge = tuple(int(c * 0.72) for c in col)
    rend.append((depth, [(q[0], q[1]) for q in pp], col + (255,), edge + (255,)))
rend.sort(key=lambda r: -r[0])
for _, pp, col, edge in rend:
    drw.polygon(pp, fill=col, outline=edge, width=2)
img = img.resize((1500, 2000), Image.LANCZOS)
img.save('render_ifc.png')

# ---------------- grade + composite ----------------
sp = np.array(img).astype(np.float32)
rgb, al = sp[..., :3], sp[..., 3] / 255.0
rgb *= np.array([0.93, 0.935, 0.945])
gray = rgb.mean(axis=2, keepdims=True)
rgb = rgb * 0.93 + gray * 0.07
rng = np.random.default_rng(11)
rgb += rng.normal(0, 2.4, rgb.shape)
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

feet = {}
for key, g in GROUND.items():
    u, v, _ = proj((key[0], key[1], g))
    feet[key] = (u / SS, v / SS)
print("Pfostenfuesse:", {k: (round(u), round(v)) for k, (u, v) in feet.items()})

band = []
steps = 12
for i in range(steps + 1):
    t = i / steps
    band.append(('poly', [(252, 838 + 140 * t), (1127, 875 + 140 * t),
                          (1127, 875 + 140 * (t + 1 / steps)), (252, 838 + 140 * (t + 1 / steps))],
                 int(200 * (1 - t))))
wall = draw_mask(band, 22) * 0.16
ground = draw_mask([('ellipse', (220, 1170, 1160, 1300), 255)], 45) * 0.11
contact = draw_mask(
    [('ellipse', (u - 40, v - 10, u + 40, v + 10), 255) for u, v in feet.values()], 9) * 0.26
shade = np.clip(wall + ground + contact, 0, 0.4)
base *= (1 - shade[..., None])
out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
out.paste(sprite, (0, 0), sprite)
out.save('final_ifc.jpg', quality=93)
print("final_ifc.jpg gespeichert")
