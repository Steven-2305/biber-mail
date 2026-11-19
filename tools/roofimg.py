#!/usr/bin/env python3
"""Dachbild aus amtlichen Luftbildern (Open Data). Niedersachsen: LGLN STAC/COG (braucht rasterio, pyproj). Aufruf: roofimg.py "<Straße>, <PLZ Ort>" out.jpg"""
import sys, json, math, io, urllib.request, urllib.parse
from PIL import Image, ImageDraw, ImageFont
UA = {'User-Agent': 'BiberPV-Angebotsrechner/1.0'}
def get(url, t=30): return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t)
def geocode(addr):
    q = urllib.parse.urlencode({'format': 'json', 'limit': 1, 'countrycodes': 'de', 'addressdetails': 1, 'q': addr})
    j = json.load(get('https://nominatim.openstreetmap.org/search?' + q))
    return (float(j[0]['lat']), float(j[0]['lon']), j[0].get('address', {}).get('state', '')) if j else None
SRC = {
 'Niedersachsen': ('https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms', 'ni_dop20', '© LGLN 2026, dl-de/by-2-0'),
 'Sachsen-Anhalt': ('https://www.geodatenportal.sachsen-anhalt.de/wss/service/ST_LVermGeo_DOP_WMS_OpenData/guest', 'lsa_lvermgeo_dop20_2', '© GeoBasis-DE / LVermGeo ST, dl-de/by-2-0'),
}
def render(lat, lon, state, out, W=1000, H=700, span=70):
    if state not in SRC: return {'ok': False, 'err': 'kein Luftbild-Dienst für ' + (state or '?')}
    url, layer, attr = SRC[state]; R = 6378137.0
    x, y = R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    hw = span / math.cos(math.radians(lat)) / 2; hh = hw * H / W
    p = {'SERVICE': 'WMS', 'VERSION': '1.3.0', 'REQUEST': 'GetMap', 'LAYERS': layer, 'STYLES': '', 'CRS': 'EPSG:3857', 'BBOX': f'{x-hw},{y-hh},{x+hw},{y+hh}', 'WIDTH': W, 'HEIGHT': H, 'FORMAT': 'image/jpeg'}
    r = get(url + '?' + urllib.parse.urlencode(p), 60); data = r.read()
    if not r.headers.get('Content-Type', '').startswith('image'): return {'ok': False, 'err': 'Luftbild-Dienst liefert gerade kein Bild (Wartung?)', 'state': state}
    im = Image.open(io.BytesIO(data)).convert('RGB'); d = ImageDraw.Draw(im); cx, cy = W // 2, H // 2
    d.ellipse([cx-11, cy-30, cx+11, cy-8], fill=(190, 0, 13), outline='white', width=2); d.polygon([(cx-8, cy-14), (cx+8, cy-14), (cx, cy)], fill=(190, 0, 13)); d.ellipse([cx-4, cy-23, cx+4, cy-15], fill='white')
    try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception: f = ImageFont.load_default()
    tw = d.textlength(attr, font=f); d.rectangle([W-tw-12, H-22, W, H], fill='white'); d.text((W-tw-6, H-19), attr, fill=(60, 60, 60), font=f)
    im.save(out, 'JPEG', quality=70, optimize=True); return {'ok': True, 'src': attr, 'state': state, 'lat': lat, 'lon': lon}

def render_ni(lat, lon, out, W=1000, H=700, span=70):
    import rasterio, numpy as np
    from rasterio.merge import merge
    from pyproj import Transformer
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:25832', always_xy=True).transform(lon, lat)
    hw = span / 2; hh = hw * H / W; b = (x - hw, y - hh, x + hw, y + hh)
    d = 0.002; q = urllib.parse.urlencode({'bbox': f'{lon-d},{lat-d},{lon+d},{lat+d}', 'limit': 50})
    feats = json.load(get('https://dop.stac.lgln.niedersachsen.de/collections/DOP/items?' + q, 40)).get('features', [])
    if not feats: return {'ok': False, 'err': 'kein Luftbild gefunden'}
    last = max(f['properties']['datetime'] for f in feats)
    urls = [f['assets']['dop20_rgb']['href'] for f in feats if f['properties']['datetime'] == last]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        srcs = [rasterio.open('/vsicurl/' + u) for u in urls]
        arr, _ = merge(srcs, bounds=b, res=0.2)
        for s_ in srcs: s_.close()
    im = Image.fromarray(np.moveaxis(arr[:3], 0, -1).astype('uint8')).resize((W, H), Image.LANCZOS)
    return finish(im, out, f'© LGLN {last[:4]}, dl-de/by-2-0', W, H, {'state': 'Niedersachsen', 'lat': lat, 'lon': lon, 'date': last[:10]})
def finish(im, out, attr, W, H, meta):
    d = ImageDraw.Draw(im); cx, cy = W // 2, H // 2
    d.ellipse([cx-11, cy-30, cx+11, cy-8], fill=(190, 0, 13), outline='white', width=2); d.polygon([(cx-8, cy-14), (cx+8, cy-14), (cx, cy)], fill=(190, 0, 13)); d.ellipse([cx-4, cy-23, cx+4, cy-15], fill='white')
    try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception: f = ImageFont.load_default()
    tw = d.textlength(attr, font=f); d.rectangle([W-tw-12, H-22, W, H], fill='white'); d.text((W-tw-6, H-19), attr, fill=(60, 60, 60), font=f)
    im.save(out, 'JPEG', quality=70, optimize=True); return {'ok': True, 'src': attr, **meta}

if __name__ == '__main__':
    g = geocode(sys.argv[1])
    print(json.dumps((render_ni(g[0], g[1], sys.argv[2]) if g[2] == 'Niedersachsen' else render(*g, sys.argv[2])) if g else {'ok': False, 'err': 'Adresse nicht gefunden'}, ensure_ascii=False))
