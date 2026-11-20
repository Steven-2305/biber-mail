#!/usr/bin/env python3
"""3D-Ansicht: Haus (LGLN LoD2) mit Luftbild-Textur und PV-Modulen.
render3d.py <lat> <lon> <raw.jpg> <mods.json> <out.jpg>"""
import sys, json, math, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate
from pyproj import Transformer
from lxml import etree
import roofpv as RP
warnings.filterwarnings('ignore')
OW, OH, SS = 1400, 950, 2           # Ausgabegröße, Supersampling
SPAN, IW, IH = 70.0, 1000, 700
def surfaces(gml):
    root = etree.fromstring(gml); ln = lambda e: etree.QName(e).localname if isinstance(e.tag, str) else ''
    for b in root.iter():
        if ln(b) != 'Building': continue
        out = []
        for s in b.iter():
            k = ln(s)
            if k in ('RoofSurface', 'WallSurface', 'GroundSurface'):
                for p in s.iter():
                    if ln(p) == 'posList': out.append((k, RP.pts(p)))
        if out: yield b.get('{http://www.opengis.net/gml}id'), out
def tris(P):
    P = P[:-1] if np.allclose(P[0], P[-1]) else P
    if len(P) < 3: return []
    # Newell-Normale: robust auch bei (fast) kollinearen ersten Punkten
    c0 = P.mean(0); Q = P - c0; n = np.zeros(3)
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]; n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    if np.linalg.norm(n) < 1e-9: return []
    n /= np.linalg.norm(n)
    u = np.cross([0, 0, 1.0], n)
    if np.linalg.norm(u) < 1e-6: u = np.array([1.0, 0, 0])
    u /= np.linalg.norm(u); v = np.cross(n, u)
    q = [(p @ u, p @ v) for p in Q]
    poly = Polygon(q).buffer(0)
    if poly.is_empty: return []
    out = []
    for t in triangulate(poly):
        if poly.contains(t.representative_point()):
            out.append(np.array([c0 + u * a + v * b for a, b in list(t.exterior.coords)[:3]]))
    return out
def main(lat, lon, raw, modsf, out):
    M = json.load(open(modsf)); X, Y, bid = M['X'], M['Y'], M['bid']
    tex = np.asarray(Image.open(raw).convert('RGB')).astype(np.float32)
    def texsample(x, y):
        c = ((x - (X - SPAN / 2)) / SPAN * IW).astype(int); r = (((Y + SPAN * IH / IW / 2) - y) / (SPAN * IH / IW) * IH).astype(int)
        ok = (c >= 0) & (c < IW) & (r >= 0) & (r < IH); res = np.full(x.shape + (3,), 205.0)
        res[ok] = tex[r[ok], c[ok]]; return res, ok
    # Szene
    items = []  # (art, tri[3x3], farbe/None, gebäude)
    zmin = None; main_c = None
    for u in RP.tile(lat, lon):
        for b, surfs in surfaces(RP.get(u)):
            allp = np.vstack([p for _, p in surfs]); c = allp[:, :2].mean(0)
            if np.hypot(c[0] - X, c[1] - Y) > 32: continue
            me = b == bid
            if me: zmin = allp[:, 2].min(); main_c = c; mainpts = allp
            for k, P in surfs:
                for t in tris(P): items.append((k, t, me, c))
    if zmin is None: raise SystemExit('Gebäude nicht gefunden')
    # Kamera: blickt auf die PV-Seite (Normalenmittel der nicht-Nord-Module), von schräg oben
    ns = [m['n'] for m in M['mods'] if not m['nord']] or [m['n'] for m in M['mods']] or [[0, -1, 1]]
    nm = np.mean(ns, 0); hd = np.array([nm[0], nm[1]]); hd = hd / (np.linalg.norm(hd) or 1)
    rot = math.radians(28); hd = np.array([hd[0] * math.cos(rot) - hd[1] * math.sin(rot), hd[0] * math.sin(rot) + hd[1] * math.cos(rot)])
    tgt = np.array([main_c[0], main_c[1], zmin + (mainpts[:, 2].max() - zmin) * 0.45])
    ext = np.ptp(mainpts[:, :2], 0).max(); dist = max(34, ext * 3.4)
    cam = tgt + np.array([hd[0] * dist * math.cos(math.radians(46)), hd[1] * dist * math.cos(math.radians(46)), dist * math.sin(math.radians(46))])
    fwd = tgt - cam; fwd /= np.linalg.norm(fwd); rgt = np.cross(fwd, [0, 0, 1]); rgt /= np.linalg.norm(rgt); upv = np.cross(rgt, fwd)
    W, H = OW * SS, OH * SS; f = W / (2 * math.tan(math.radians(22)))
    def proj(P):
        d = P - cam; z = d @ fwd; return np.stack([W / 2 + f * (d @ rgt) / z, H / 2 - f * (d @ upv) / z], -1), z
    img = np.zeros((H, W, 3), np.float32); zb = np.full((H, W), np.inf, np.float32)
    # Himmel
    yy = np.linspace(0, 1, H)[:, None]; img[:] = (np.array([150, 190, 235]) * (1 - yy) + np.array([225, 236, 248]) * yy)[:, None, :][:, 0][:, None, :]
    # Boden (Luftbild) über Strahl-Ebenen-Schnitt
    jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
    dirs = fwd[None, None, :] + ((ii - W / 2) / f)[..., None] * rgt + (-(jj - H / 2) / f)[..., None] * upv
    t = (zmin - cam[2]) / np.where(np.abs(dirs[..., 2]) < 1e-6, -1e-6, dirs[..., 2]); g = t > 0
    gx, gy = cam[0] + t * dirs[..., 0], cam[1] + t * dirs[..., 1]
    col, ok = texsample(gx[g], gy[g]); dd = np.hypot(gx[g] - X, gy[g] - Y)
    fade = np.clip((dd - 22) / 12, 0, 1)[:, None]; base = col * (1 - fade) + np.array([178, 186, 172]) * fade
    img[g] = base; zb[g] = (t[g] * (dirs[g] @ fwd)) + 0.5
    sun = np.array([-0.45, -0.6, 0.66]); sun /= np.linalg.norm(sun)
    def raster(T, shade_fn):
        p, z = proj(T)
        if (z < 0.5).any(): return
        x0, x1 = int(max(0, p[:, 0].min())), int(min(W - 1, p[:, 0].max() + 1)); y0, y1 = int(max(0, p[:, 1].min())), int(min(H - 1, p[:, 1].max() + 1))
        if x0 >= x1 or y0 >= y1: return
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32); (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9: return
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den; l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den; l3 = 1 - l1 - l2
        m = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
        if not m.any(): return
        iz = l1 / z[0] + l2 / z[1] + l3 / z[2]; zz = 1 / iz
        sub = zb[y0:y1 + 1, x0:x1 + 1]; m &= zz < sub
        if not m.any(): return
        w = np.stack([l1 / z[0], l2 / z[1], l3 / z[2]], -1) * zz[..., None]
        wp = w[m] @ T
        sub[m] = zz[m]; img[y0:y1 + 1, x0:x1 + 1][m] = shade_fn(wp)
    for k, T, me, bc in items:
        if not me and ((bc - tgt[:2]) @ hd > 3 or np.hypot(*(bc - tgt[:2])) > 26): continue   # Nachbarn vor dem Haus nicht zeichnen (verdecken sonst)
        n = np.cross(T[1] - T[0], T[2] - T[0]); n /= (np.linalg.norm(n) or 1)
        if k == 'RoofSurface' and n[2] < 0: n = -n
        if k == 'WallSurface' and ((T.mean(0) - cam) @ n) > 0: n = -n
        lam = 0.55 + 0.45 * max(0.0, float(n @ sun))
        if k == 'RoofSurface':
            cc, _ = texsample(T[:, 0], T[:, 1]); med = np.clip(np.median(cc, 0), 60, 200)
            tone = np.array([150, 70, 55]) if med[0] > med[2] + 18 else np.array([88, 92, 98])   # Ziegelrot oder Anthrazit
            col = tone * (0.72 + 0.4 * lam) if me else tone * 0.6 * (0.72 + 0.4 * lam) + 80
            def sh(wp, col=col, T=T):
                k_ = ((wp - T[0]) @ np.array([0, 0, 1.0]))
                rows = (np.abs((k_ * 3.2) % 1 - 0.5) > 0.44)
                c = np.tile(col, (len(wp), 1)); c[rows] *= 0.9; return c
        elif k == 'WallSurface':
            base = np.array([238, 232, 222]) if me else np.array([210, 210, 208])
            def sh(wp, lam=lam, base=base): return np.tile(base * (0.62 + 0.38 * lam), (len(wp), 1))
        else: continue
        raster(T, sh)
    # Module
    edges = []
    for mo in M['mods']:
        C = np.array(mo['c']); n = np.array(mo['n']); spec = max(0.0, float(n @ sun))
        base = np.array([28, 38, 66]) if not mo['nord'] else np.array([40, 50, 78])
        def shm(wp, C=C, base=base, spec=spec):
            a = (wp - C[0]) @ (C[1] - C[0]) / np.dot(C[1] - C[0], C[1] - C[0]); b = (wp - C[0]) @ (C[3] - C[0]) / np.dot(C[3] - C[0], C[3] - C[0])
            ln_ = (np.abs((a * 6) % 1 - 0.5) > 0.47) | (np.abs((b * 10) % 1 - 0.5) > 0.47)
            frame = (a < 0.025) | (a > 0.975) | (b < 0.018) | (b > 0.982)
            c = np.tile(base + 45 * spec, (len(wp), 1)) + (40 * (1 - b))[:, None] * np.array([0.6, 0.7, 1.0])
            c[ln_] = c[ln_] * 0.75 + 12; c[frame] = [24, 26, 30]
            return c
        raster(C[[0, 1, 2]], shm); raster(C[[0, 2, 3]], shm)
    for Q in M.get('obs', []):
        Q = np.array(Q)
        def sho(wp): return np.tile(np.array([70, 90, 110.0]), (len(wp), 1))
        raster(Q[[0, 1, 2]], sho); raster(Q[[0, 2, 3]], sho)
    # leichte Konturen über Tiefensprünge
    zl = np.where(np.isinf(zb), 1e4, zb); e = np.zeros_like(zl, bool)
    e[1:, :] |= np.abs(np.diff(zl, axis=0)) > 0.35 * zl[1:, :] * 0.04; e[:, 1:] |= np.abs(np.diff(zl, axis=1)) > 0.35 * zl[:, 1:] * 0.04
    img[e] = img[e] * 0.55
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((OW, OH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    try: fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception: fb = ImageFont.load_default()
    a = 'Visualisierung · 3D: © LGLN, Luftbild: © LGLN 2025, dl-de/by-2-0'; tw = d.textlength(a, font=fb)
    d.rectangle([OW - tw - 16, OH - 26, OW, OH], fill=(255, 255, 255)); d.text((OW - tw - 8, OH - 22), a, fill=(80, 80, 80), font=fb)
    im.save(out, 'JPEG', quality=86)
if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
