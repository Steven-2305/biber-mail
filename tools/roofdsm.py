#!/usr/bin/env python3
"""Ersatz-Dachmodell aus dem bildbasierten Oberflächenmodell (LGLN bDOM20, gleiche Befliegung wie das Luftbild),
wenn das Haus noch nicht im LoD2 ist. Liefert Dachflächen (3D-Polygone) und Wände wie aus LoD2.
roofdsm.py <lat> <lon> <out.json> [debug.png]"""
import sys, json, math, warnings
import numpy as np, rasterio
from rasterio.merge import merge
from pyproj import Transformer
from scipy import ndimage as ndi
from skimage import measure, morphology
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import urllib.request
warnings.filterwarnings('ignore')
RES = 0.2
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'BiberPV-Angebotsrechner/1.0'}), timeout=60).read()
def bdom(lat, lon, X, Y, half=30):
    d = 0.0006; feats = json.loads(get(f'https://bdom.stac.lgln.niedersachsen.de/collections/BDOM/items?bbox={lon-d},{lat-d},{lon+d},{lat+d}&limit=20'))['features']
    last = max(f['properties']['datetime'] for f in feats); urls = [f['assets']['bdom20']['href'] for f in feats if f['properties']['datetime'] == last]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        srcs = [rasterio.open('/vsicurl/' + u) for u in urls]; arr, tr = merge(srcs, bounds=(X - half, Y - half, X + half, Y + half), res=RES); nod = srcs[0].nodata
        for s in srcs: s.close()
    z = arr[0].astype(float)
    if nod is not None: z[z == nod] = np.nan
    return z, tr, last[:10]
def planes(z, mask):
    zs = ndi.gaussian_filter(np.nan_to_num(z, nan=np.nanmedian(z)), 1.2)
    gy, gx = np.gradient(zs, RES); gy = -gy                      # Zeilen laufen nach Süden
    n = np.dstack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    tilt = np.degrees(np.arccos(n[..., 2])); az = (np.degrees(np.arctan2(n[..., 0], n[..., 1])) + 360) % 360
    # Klassen: flach oder 16 Ausrichtungssektoren
    cls = np.where(tilt < 8, 100, (((az + 11.25) % 360) // 22.5).astype(int))
    cls[~mask] = -1; cls[(tilt > 65) & mask] = -2               # Kanten/Wände
    regs = []
    for c in np.unique(cls[cls >= 0]):
        lab = measure.label(morphology.binary_opening(cls == c, morphology.disk(2)))
        for rp in measure.regionprops(lab):
            if rp.area * RES * RES < 4: continue
            rr, cc = rp.coords[:, 0], rp.coords[:, 1]; A = np.c_[cc * RES, -rr * RES, np.ones(len(rr))]; zz = z[rr, cc]; ok = np.isfinite(zz)
            if ok.sum() < 30: continue
            coef, *_ = np.linalg.lstsq(A[ok], zz[ok], rcond=None)
            regs.append({'mask': lab == rp.label, 'coef': coef, 'flat': c == 100})
    # benachbarte Regionen mit gleicher Ebene zusammenführen (Sektor-Grenzen)
    def nrm(cf): v = np.array([-cf[0], -cf[1], 1.0]); return v / np.linalg.norm(v)
    merged = True
    while merged:
        merged = False
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                if a['flat'] != b['flat']: continue
                if np.degrees(np.arccos(min(1, nrm(a['coef']) @ nrm(b['coef'])))) > 9: continue
                if not (morphology.binary_dilation(a['mask'], morphology.disk(2)) & b['mask']).any(): continue
                m = a['mask'] | b['mask']; rr, cc = np.nonzero(m); A = np.c_[cc * RES, -rr * RES, np.ones(len(rr))]; zz = z[rr, cc]; ok = np.isfinite(zz)
                coef, *_ = np.linalg.lstsq(A[ok], zz[ok], rcond=None); regs[i] = {'mask': m, 'coef': coef, 'flat': a['flat']}; del regs[j]; merged = True; break
            if merged: break
    return regs
def poly_of(mask, tr):
    cs = measure.find_contours(np.pad(mask, 1).astype(float), 0.5); polys = []
    for c in cs:
        pts = [(tr.c + (x - 1) * RES, tr.f - (y - 1) * RES) for y, x in c]
        if len(pts) >= 4: polys.append(Polygon(pts).buffer(0))
    if not polys: return None
    p = unary_union(polys); p = max(p.geoms, key=lambda g: g.area) if isinstance(p, MultiPolygon) else p
    return p.simplify(0.35)
def main(lat, lon, out, dbg=None):
    X, Y = Transformer.from_crs('EPSG:4326', 'EPSG:25832', always_xy=True).transform(lon, lat)
    z, tr, date = bdom(lat, lon, X, Y)
    ground = np.nanpercentile(z, 4); nd = z - ground
    bld = np.nan_to_num(nd) > 2.6
    bld = morphology.binary_opening(bld, morphology.disk(3))
    lab = measure.label(bld); r0, c0 = int((tr.f - Y) / RES), int((X - tr.c) / RES)
    best, bd = None, 1e9
    for rp in measure.regionprops(lab):
        if rp.area * RES * RES < 30: continue
        d_ = np.min(np.hypot(rp.coords[:, 0] - r0, rp.coords[:, 1] - c0)) * RES
        if d_ < bd: bd, best = d_, rp.label
    if best is None or bd > 8: return {'ok': False, 'err': 'kein Gebäude im Oberflächenmodell gefunden'}
    mask = morphology.binary_closing(lab == best, morphology.disk(2))
    fp0 = poly_of(mask, tr)
    if fp0 is not None:
        c_ = list(fp0.minimum_rotated_rectangle.exterior.coords); L_ = max(math.dist(c_[0], c_[1]), math.dist(c_[1], c_[2]))
        if fp0.area > 350 or L_ > 30: return {'ok': False, 'err': 'Reihen- oder Mehrfamilienhaus – die eigene Dachfläche lässt sich ohne amtliches 3D-Modell nicht sicher abgrenzen. Belegung bitte vor Ort schätzen.'}
    regs = planes(z, mask)
    roofs = []; fp = poly_of(mask, tr)
    # Ausrichtung an den Gebäudekanten einrasten (Grundriss als gedrehtes Rechteck)
    ang = None
    if fp is not None:
        mrr = list(fp.minimum_rotated_rectangle.exterior.coords); ex, ey = mrr[1][0] - mrr[0][0], mrr[1][1] - mrr[0][1]; ang = math.degrees(math.atan2(ex, ey)) % 90
    for r in regs:
        if r['flat']: continue
        pr = poly_of(r['mask'], tr)
        if pr is not None and pr.area > 8:
            mrr = list(pr.minimum_rotated_rectangle.exterior.coords); ex, ey = mrr[1][0] - mrr[0][0], mrr[1][1] - mrr[0][1]; ang = math.degrees(math.atan2(ex, ey)) % 90
        if ang is None: continue
        a0, b0, _ = r['coef']; az = (math.degrees(math.atan2(-a0, -b0)) + 360) % 360
        cands = [(ang + k * 90) % 360 for k in range(4)]; snap = min(cands, key=lambda c: abs((c - az + 180) % 360 - 180))
        if abs((snap - az + 180) % 360 - 180) > 20: continue
        rr, cc = np.nonzero(r['mask']); x_, y_ = cc * RES, -rr * RES; zz = z[rr, cc]; ok = np.isfinite(zz)
        dx, dy = math.sin(math.radians(snap)), math.cos(math.radians(snap)); t_ = x_ * dx + y_ * dy
        A = np.c_[t_, np.ones(len(t_))]; (sl, c0), *_ = np.linalg.lstsq(A[ok], zz[ok], rcond=None)
        r['coef'] = np.array([sl * dx, sl * dy, c0])
    for r in regs:
        p = poly_of(r['mask'], tr)
        if p is None or p.area < 4: continue
        if fp is not None:
            q = p.buffer(0.45, join_style=2).intersection(fp.buffer(0.1))
            q = max(q.geoms, key=lambda g: g.area) if hasattr(q, 'geoms') else q
            if not q.is_empty and q.area > p.area: p = q.simplify(0.25)
        a, b, c = r['coef']
        zf = lambda x, y: a * (x - tr.c) + b * (y - tr.f) + c
        ring = [(x, y, zf(x, y)) for x, y in list(p.exterior.coords)]
        roofs.append(ring)
    # Restflächen des Grundrisses (z. B. verschattete Dachseiten, die das Oberflächenmodell schlecht trifft) als eigene Ebenen ergänzen
    if fp is not None:
        from shapely.geometry import Polygon as _Pg
        cov = unary_union([_Pg([(x, y) for x, y, _ in r]).buffer(0.15) for r in roofs]) if roofs else None
        rest = fp.difference(cov) if cov is not None else fp
        for g in (rest.geoms if hasattr(rest, 'geoms') else [rest]):
            g = g.buffer(-0.3).buffer(0.3)
            if g.is_empty or g.area < 8: continue
            g = max(g.geoms, key=lambda q: q.area) if hasattr(g, 'geoms') else g
            from rasterio.features import rasterize
            m_ = rasterize([g], out_shape=z.shape, transform=tr).astype(bool)
            rr, cc = np.nonzero(m_); A = np.c_[cc * RES, -rr * RES, np.ones(len(rr))]; zz = z[rr, cc]; ok = np.isfinite(zz)
            if ok.sum() < 50: continue
            (a1, b1, c1), *_ = np.linalg.lstsq(A[ok], zz[ok], rcond=None)
            zf2 = lambda x, y, a1=a1, b1=b1, c1=c1: a1 * (x - tr.c) + b1 * (y - tr.f) + c1
            gs = g.simplify(0.3); roofs.append([(x, y, zf2(x, y)) for x, y in list(gs.exterior.coords)])
    walls = []
    if fp is not None:
        ring = list(fp.exterior.coords)
        for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
            def top(x, y):
                best_ = None
                for r_ in roofs:
                    P_ = np.array(r_); pg = Polygon(P_[:, :2]).buffer(0)
                    d_ = pg.distance(Point(x, y))
                    if d_ < 1.2 and (best_ is None or d_ < best_[0]):
                        A_ = np.c_[P_[:, 0], P_[:, 1], np.ones(len(P_))]; cf, *_ = np.linalg.lstsq(A_, P_[:, 2], rcond=None); best_ = (d_, cf @ [x, y, 1])
                if best_: return float(best_[1])
                rr, cc = int((tr.f - y) / RES), int((x - tr.c) / RES); w = nd[max(0, rr - 3):rr + 4, max(0, cc - 3):cc + 4]
                return ground + float(np.nanpercentile(w, 60)) if np.isfinite(w).any() else ground + 3
            walls.append([(x1, y1, ground), (x2, y2, ground), (x2, y2, top(x2, y2)), (x1, y1, top(x1, y1))])
    json.dump({'src': 'bdom', 'date': date, 'roofs': roofs, 'walls': walls, 'ground': ground, 'footprint_m2': round(fp.area, 1) if fp else 0}, open(out, 'w'))
    if dbg:
        from PIL import Image
        im = np.zeros(z.shape + (3,), np.uint8); im[..., :] = (np.clip(np.nan_to_num(nd), 0, 12) / 12 * 255).astype(np.uint8)[..., None]
        rng = np.random.default_rng(1)
        for r in regs: im[r['mask']] = rng.integers(60, 255, 3)
        Image.fromarray(im).save(dbg)
    return {'ok': True, 'faces': len(roofs), 'date': date}
if __name__ == '__main__':
    print(json.dumps(main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)))
