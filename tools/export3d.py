#!/usr/bin/env python3
"""Exportiert die 3D-Szene (LoD2 + Module + Störflächen) als kompaktes JSON für die Ansicht im Rechner.
export3d.py <lat> <lon> <raw.jpg> <mods.json> <out.json>"""
import sys, json, numpy as np
from PIL import Image
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
    if zmin is None: raise SystemExit('Gebäude nicht gefunden')
    loc = lambda a: [round(a[i] - (X if i % 3 == 0 else Y if i % 3 == 1 else zmin), 2) for i in range(len(a))]
    for b in blds:
        for r in b['r']: r['v'] = loc(r['v'])
        b['w'] = loc(b['w']); b['gz'] = round(b['gz'] - zmin, 2)
    mods = []
    for i, m in enumerate(M['mods']):
        mods.append({'c': loc([x for p in m['c'] for x in p]), 'o': 'N' if m['nord'] else m.get('o', 'S')})
    obs = [loc([x for p in q for x in p]) for q in M.get('obs', [])]
    json.dump({'v': 1, 'b': blds, 'm': mods, 'obs': obs, 'g': [SPAN, SPAN * IH / IW]}, open(out, 'w'), separators=(',', ':'))
if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
