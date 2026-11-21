#!/usr/bin/env python3
"""Ergänzt eine vorhandene 3D-Szene (<id>-3d.json) um die Dachflächen-Rahmen des Kundenhauses.
faces3d.py <lat> <lon> <3d.json>
Je Dachfläche: o (Himmelsrichtung wie im Rechner), tilt, az, O (Traufe links, lokale Koordinaten),
e (Traufrichtung nach rechts, von außen auf die Fläche gesehen), u (hangaufwärts in der Ebene), n, W (Traufbreite), L (Traufe->First), A (Fläche m²)."""
import sys, json, math, numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pyproj import Transformer
import roofpv as RP, render3d as R3
def main(lat, lon, path):
    D = json.load(open(path))
    X, Y = Transformer.from_crs('EPSG:4326', 'EPSG:25832', always_xy=True).transform(lon, lat); p = Point(X, Y); cand = []
    for u in RP.tile(lat, lon):
        g = RP.get(u)
        for bid, roofs in RP.buildings(g):
            fp = unary_union([Polygon(r[:, :2]).buffer(0) for r in roofs if len(r) >= 3]); d = fp.distance(p)
            if d < 12: cand.append((d, -fp.area, bid, g))
    if not cand: raise SystemExit('Gebäude nicht gefunden')
    cand.sort(key=lambda c: (c[0], c[1])); _, _, bid, g = cand[0]
    surfs = next(s for b, s in R3.surfaces(g) if b == bid)
    zmin = float(np.vstack([q for _, q in surfs])[:, 2].min()); O0 = np.array([X, Y, zmin])
    faces = []
    for k, P in surfs:
        if k != 'RoofSurface' or len(P) < 3: continue
        P = P[:-1] if np.allclose(P[0], P[-1]) else P
        n, tilt, az, _, _ = RP.frame(P)
        if tilt < 8: continue
        f = -np.array([n[0], n[1]]); f /= np.linalg.norm(f)
        e = np.array([f[1], -f[0], 0.0]); uu = np.array([0, 0, 1.0]) - n * n[2]; uu /= np.linalg.norm(uu)
        c = P.mean(0); s = (P - c) @ e; t = (P - c) @ uu
        O = c + s.min() * e + t.min() * uu - O0
        area = Polygon(np.c_[s, t]).area
        faces.append({'o': RP.ori(az), 'tilt': round(tilt, 1), 'az': round(az), 'O': [round(x, 3) for x in O], 'e': [round(x, 4) for x in e], 'u': [round(x, 4) for x in uu], 'n': [round(x, 4) for x in n],
                      'W': round(float(s.max() - s.min()), 2), 'L': round(float(t.max() - t.min()), 2), 'A': round(float(area), 1)})
    faces.sort(key=lambda x: -x['A'])
    D['f'] = faces; json.dump(D, open(path, 'w'), separators=(',', ':'))
    print(json.dumps([{k: v for k, v in x.items() if k in ('o', 'tilt', 'W', 'L', 'A')} for x in faces]))
if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
