#!/usr/bin/env python3
"""Exportiert die 3D-Szene (LoD2 + Module + Störflächen) als kompaktes JSON für die Ansicht im Rechner.
export3d.py <lat> <lon> <raw.jpg> <mods.json> <out.json>"""
import sys, json, numpy as np
from PIL import Image
from shapely.geometry import Polygon
import roofpv as RP, render3d as R3
SPAN, IW, IH = 70.0, 1000, 700
def main(lat, lon, raw, modsf, out):
    M = json.load(open(modsf)); X, Y, bid = M['X'], M['Y'], M['bid']
    tex = np.asarray(Image.open(raw).convert('RGB')).astype(float)
    def samp(P):
        c = np.clip(((P[:, 0] - (X - SPAN / 2)) / SPAN * IW).astype(int), 0, IW - 1); r = np.clip((((Y + SPAN * IH / IW / 2) - P[:, 1]) / (SPAN * IH / IW) * IH).astype(int), 0, IH - 1)
        return tex[r, c]
    blds = []; zmin = None
    for u in RP.tile(lat, lon):
        for b, surfs in R3.surfaces(RP.get(u)):
            allp = np.vstack([p for _, p in surfs]); c = allp[:, :2].mean(0)
            if np.hypot(c[0] - X, c[1] - Y) > 30: continue
            me = b == bid
            if me: zmin = float(allp[:, 2].min())
            roofs, walls = [], []
            for k, P in surfs:
                ts = R3.tris(P)
                if not ts: continue
                if k == 'RoofSurface':
                    med = np.median(samp(np.vstack(ts)), 0); tone = 'r' if med[0] > med[2] + 18 else 'g'
                    roofs.append({'t': tone, 'v': [x for T in ts for p in T for x in p]})
                elif k == 'WallSurface':
                    walls += [x for T in ts for p in T for x in p]
            blds.append({'me': me, 'r': roofs, 'w': walls, 'gz': float(allp[:, 2].min())})
    if bid == 'dsm':
        import os
        dsm = json.load(open(os.environ['ROOF_DSM'])); zmin = float(dsm['ground'])
        roofs = []
        for r in dsm['roofs']:
            ts = R3.tris(np.array(r))
            if ts: med = np.median(samp(np.vstack(ts)), 0); roofs.append({'t': 'r' if med[0] > med[2] + 18 else 'g', 'v': [x for T in ts for p in T for x in p]})
        walls = [x for w in dsm['walls'] for T in R3.tris(np.array(w)) for p in T for x in p]
        blds = [b for b in blds if not Polygon([(0, 0), (1, 0), (1, 1)]).is_empty]
        fpc = np.array([p for w in dsm['walls'] for p in w])[:, :2].mean(0)
        blds = [b for b in blds if not b['me']]
        blds.append({'me': True, 'r': roofs, 'w': walls, 'gz': zmin})
    if zmin is None: raise SystemExit('Gebäude nicht gefunden')
    loc = lambda a: [round(a[i] - (X if i % 3 == 0 else Y if i % 3 == 1 else zmin), 2) for i in range(len(a))]
    for b in blds:
        for r in b['r']: r['v'] = loc(r['v'])
        b['w'] = loc(b['w']); b['gz'] = round(b['gz'] - zmin, 2)
    mods = []
    for i, m in enumerate(M['mods']):
        mods.append({'c': loc([x for p in m['c'] for x in p]), 'o': 'N' if m['nord'] else m.get('o', 'S')})
    obs = [loc([x for p in q for x in p]) for q in M.get('obs', [])]
    # Selbstprüfung: jedes Modul muss knapp ÜBER einer Dachfläche des Hauses liegen
    me = [b for b in blds if b['me']][0]; planes = []
    for f in me['r']:
        v = np.array(f['v']).reshape(-1, 3, 3)
        for T in v:
            n = np.cross(T[1] - T[0], T[2] - T[0]); 
            if np.linalg.norm(n) < 1e-9: continue
            n = n / np.linalg.norm(n) * np.sign(n[2] or 1); planes.append((T, n))
    from shapely.geometry import Polygon as _P, Point as _Pt
    bad = 0
    for m in mods:
        c = np.array(m['c']).reshape(-1, 3).mean(0); best = None
        for T, n in planes:
            if _P(T[:, :2]).buffer(0.05).contains(_Pt(c[0], c[1])):
                d_ = float((c - T[0]) @ n); best = d_ if best is None or abs(d_ - 0.12) < abs(best - 0.12) else best
        if best is None or not (0.0 <= best <= 0.3): bad += 1
    if bad: raise SystemExit(f'Prüfung fehlgeschlagen: {bad} Module liegen nicht auf dem Dach')
    json.dump({'v': 1, 'b': blds, 'm': mods, 'obs': obs, 'g': [SPAN, SPAN * IH / IW]}, open(out, 'w'), separators=(',', ':'))
if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
