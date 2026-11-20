#!/usr/bin/env python3
"""Mögliche maximale Modulbelegung aus LGLN-LoD2 (Niedersachsen) schätzen und auf das Dachbild zeichnen.
roofpv.py <lat> <lon> <bild.jpg> <out.jpg>   (Bild aus roofimg.py: 1000x700, 70 m breit, zentriert auf lat/lon)"""
import sys, json, math, urllib.request, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from lxml import etree
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont
from skimage import color, draw as skd, morphology
MPP = SPAN_DEFAULT = None
NS = {'gml': 'http://www.opengis.net/gml', 'bldg': 'http://www.opengis.net/citygml/building/2.0'}
MOD = (1.134, 1.80)          # Modulbreite x -länge (m), konservativ (passt für 460 W und 480 W)
EDGE, GAP = 0.30, 0.02        # Randabstand zu Traufe/First/Ortgang, Modulfuge
W, H, SPAN = 1000, 700, 70.0
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'BiberPV-Angebotsrechner/1.0'}), timeout=90).read()
def tile(lat, lon):
    d = 0.0003; q = f'bbox={lon-d},{lat-d},{lon+d},{lat+d}&limit=10'
    fs = json.loads(get('https://lod.stac.lgln.niedersachsen.de/collections/lod2/items?' + q))['features']
    return [f['assets']['lod2-gml']['href'] for f in fs]
def pts(pl):
    v = [float(x) for x in pl.text.split()]; return np.array(v).reshape(-1, 3)
def buildings(gml):
    root = etree.fromstring(gml)
    ln = lambda e: etree.QName(e).localname if isinstance(e.tag, str) else ''
    for b in root.iter():
        if ln(b) != 'Building': continue
        roofs = [pts(p) for r in b.iter() if ln(r) == 'RoofSurface' for p in r.iter() if ln(p) == 'posList']
        if roofs: yield b.get('{%s}id' % NS['gml']), roofs
def frame(P):
    n = np.zeros(3)
    for i in range(len(P)):
        a, c = P[i], P[(i + 1) % len(P)]; n += [(a[1]-c[1])*(a[2]+c[2]), (a[2]-c[2])*(a[0]+c[0]), (a[0]-c[0])*(a[1]+c[1])]
    n /= np.linalg.norm(n) or 1
    if n[2] < 0: n = -n
    tilt = math.degrees(math.acos(min(1, n[2])))
    az = (math.degrees(math.atan2(n[0], n[1])) + 360) % 360
    u = np.cross([0, 0, 1], n); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1., 0, 0])
    v = np.cross(n, u)
    return n, tilt, az, u, v
def pack(poly2d, w, h, okf=None):
    inner = poly2d.buffer(-EDGE, join_style=2)
    if inner.is_empty or inner.area < w * h: return []
    x0, y0, x1, y1 = inner.bounds; best = []
    for ox in np.arange(0, w + GAP, 0.15):
        for oy in np.arange(0, h + GAP, 0.15):
            got = []; y = y0 + oy
            while y + h <= y1 + 1e-9:
                x = x0 + ox
                while x + w <= x1 + 1e-9:
                    r = box(x, y, x + w, y + h)
                    if inner.contains(r) and (okf is None or okf(x, y, w, h)): got.append((x, y)); x += w + GAP
                    else: x += 0.1
                y += h + GAP
            if len(got) > len(best): best = got
    return best
def ori(az):
    return ['N', 'O', 'S', 'W'][int(((az + 45) % 360) // 90)]
def main(lat, lon, img, out):
    X, Y = Transformer.from_crs('EPSG:4326', 'EPSG:25832', always_xy=True).transform(lon, lat)
    p = Point(X, Y); cand = []
    for u in tile(lat, lon):
        for bid, roofs in buildings(get(u)):
            fp = unary_union([Polygon(r[:, :2]).buffer(0) for r in roofs if len(r) >= 3])
            dist = fp.distance(p)
            if dist < 12: cand.append((dist, -fp.area, bid, roofs))
    if not cand: return {'ok': False, 'err': 'kein Gebäude im 3D-Modell gefunden'}
    cand.sort(key=lambda c: (c[0], c[1])); _, _, bid, roofs = cand[0]
    raw = img[:-4] + '.raw.jpg'; A = color.rgb2lab(np.asarray(Image.open(raw if os.path.exists(raw) else img).convert('RGB')) / 255.0); im = Image.open(img).convert('RGB'); mpp = SPAN / W
    OBS = np.zeros((H, W), bool)
    def facemask(poly_px):
        m = np.zeros((H, W), bool); rr, cc = skd.polygon([p[1] for p in poly_px], [p[0] for p in poly_px], (H, W)); m[rr, cc] = True; return m
    im0 = im; ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    px = lambda x, y: ((x - (X - SPAN / 2)) / SPAN * W, ((Y + SPAN * H / W / 2) - y) / (SPAN * H / W) * H)
    faces = []; allpx = []
    for P in roofs:
        if len(P) < 4: continue
        P = P[:-1] if np.allclose(P[0], P[-1]) else P
        n, tilt, az, u, v = frame(P); o = P[0]
        q = Polygon([((pt - o) @ u, (pt - o) @ v) for pt in P]).buffer(0)
        if q.area < 6: continue
        flat = tilt < 8
        fm = facemask([px(pt[0], pt[1]) for pt in P])
        core = morphology.binary_erosion(fm, morphology.disk(max(1, int(0.45 / mpp))))
        obs = np.zeros_like(core)
        if core.sum() > 200:
            med = np.median(A[core], axis=0); dist = np.linalg.norm(A - med, axis=2)
            thr = max(16.0, np.percentile(dist[core], 80) * 2.2)
            obs = core & (dist > thr)
            obs = morphology.binary_opening(obs, morphology.disk(2))
            obs = morphology.remove_small_objects(obs, int((0.35 / mpp) ** 2))
            from skimage import measure
            lab = measure.label(obs)
            for rp in measure.regionprops(lab):
                if rp.area * mpp * mpp > 15: obs[lab == rp.label] = False   # große Flächen = Schatten/Farbverlauf, keine Störfläche
            obs = morphology.binary_dilation(obs, morphology.disk(int(0.2 / mpp)))
        OBS |= obs
        def okf(x, y, w, h, o=o, u=u, v=v, obs=obs):
            cs = [o + u * a + v * b for a, b in [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
            pp = [px(c[0], c[1]) for c in cs]; rr, cc = skd.polygon([p[1] for p in pp], [p[0] for p in pp], (H, W))
            return len(rr) == 0 or obs[rr, cc].mean() < 0.03
        best = []
        for (w, h) in [MOD, MOD[::-1]]:
            r = pack(q, w, h, okf)
            if len(r) > len(best): best, bw, bh = r, w, h
        if flat: best = best[:int(len(best) * 0.85)]   # Aufständerung Ost-West: Wartungswege
        if len(best) < 3: continue
        for (x, y) in best:
            cs = [o + u * a + v * b for a, b in [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]]
            allpx += [px(c[0], c[1]) for c in cs]
            nord = not flat and ori(az) == 'N'
            d.polygon([px(c[0], c[1]) for c in cs], fill=(70, 80, 100, 140) if nord else (20, 30, 60, 210), outline=(235, 240, 255, 200 if nord else 255))
        faces.append({'o': 'flach' if flat else ori(az), 'az': round(az), 'tilt': round(tilt), 'n': len(best), 'm2': round(q.area, 1), 'obs': int(obs.sum() * mpp * mpp * 10) / 10})
    if OBS.any():
        edge = OBS & ~morphology.binary_erosion(OBS, morphology.disk(1)); oa = np.asarray(ov).copy(); oa[OBS] = np.maximum(oa[OBS], [255, 150, 0, 70]); oa[edge] = [255, 140, 0, 230]; ov = Image.fromarray(oa)
    im = Image.alpha_composite(im.convert('RGBA'), ov).convert('RGB')
    if allpx:
        xs, ys = [a for a, _ in allpx], [b for _, b in allpx]; m = 8 / SPAN * W
        x0, x1, y0, y1 = max(0, min(xs) - m), min(W, max(xs) + m), max(0, min(ys) - m), min(H, max(ys) + m)
        cw = max(x1 - x0, (y1 - y0) * W / H); ch = cw * H / W; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = min(max(0, cx - cw / 2), W - cw), min(max(0, cy - ch / 2), H - ch)
        im = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.LANCZOS)
    tot = sum(f['n'] for f in faces); dd = ImageDraw.Draw(im)
    try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception: f = ImageFont.load_default()
    t = f'Mögliche Belegung (Schätzung): ca. {tot} Module'; tw = dd.textlength(t, font=f)
    dd.rounded_rectangle([10, 10, 30 + tw, 44], 8, fill=(190, 0, 13)); dd.text((20, 16), t, fill='white', font=f)
    im.save(out, 'JPEG', quality=72, optimize=True)
    agg = {}
    for x in faces:
        k = 'S' if x['o'] == 'flach' else x['o']; agg[k] = agg.get(k, 0) + x['n']
    return {'ok': True, 'total': tot, 'faces': faces, 'byOri': agg, 'flat': any(x['o'] == 'flach' for x in faces), 'bid': bid}
if __name__ == '__main__':
    print(json.dumps(main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]), ensure_ascii=False))
