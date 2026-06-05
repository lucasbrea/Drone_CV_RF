"""
Generate a pasture-species map with the v8 Random Forest — self-contained.

Unlike train_rf_v8.py (which needs only the bundled database), MAKING A MAP
needs the drone orthophotos: it classifies every ~30 cm patch in each image and
renders a Leaflet web map. Point --imagery at your multispectral GeoTIFF(s).

The 15 features are recomputed here exactly as the trainer's database was built
(9 spectral + 6 neighborhood stats), so the bundled rf_v8_model.pkl applies
directly. Training points are read from training_data.db and drawn on the map.

Expected imagery: 5-band orthophoto (1=Red, 2=Green, 3=NIR, 4=RedEdge,
5=alpha/mask), any GSD; patches are formed at --patch-size cm.

Run:
    pip install -r requirements.txt
    python generate_map.py --imagery "/path/to/*/odm_orthophoto.tif" --outdir map_output
"""
import argparse
import glob
import gzip
import os
import pickle
import sqlite3
from collections import Counter

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from scipy.ndimage import uniform_filter
from pyproj import Transformer
import matplotlib.pyplot as plt
import folium

BAND_RED, BAND_GREEN, BAND_NIR, BAND_REDEDGE, BAND_ALPHA = 1, 2, 3, 4, 5
HERE = os.path.dirname(os.path.abspath(__file__))

# Class display colours (RGBA 0-1). Water Hole shares Bare Soil (it was merged).
COLOR_BY_NAME = {
    'Leguminosa': (0.18, 0.63, 0.18, 1.0),
    'Graminea':   (0.50, 0.85, 0.10, 1.0),
    'Other':      (0.50, 0.50, 0.55, 1.0),
    'Bare Soil':  (0.65, 0.42, 0.18, 1.0),
    'Dry Grass':  (0.95, 0.80, 0.30, 1.0),
    'Water Hole': (0.65, 0.42, 0.18, 1.0),
}


# ---------------- feature extraction (mirrors the DB builder) ----------------

def safe_divide(num, den):
    out = np.zeros_like(num, dtype=np.float32)
    m = den != 0
    out[m] = num[m] / den[m]
    return out


def compute_indices(red, green, nir, rededge):
    ndvi = safe_divide(nir - red, nir + red)
    ndre = safe_divide(nir - rededge, nir + rededge)
    gndvi = safe_divide(nir - green, nir + green)
    gri = safe_divide(green - red, green + red)
    sr = safe_divide(nir, red + 1e-6)
    return ndvi, ndre, gndvi, gri, sr


def block_mean(arr, factor, valid_mask):
    h, w = arr.shape
    nh, nw = (h // factor) * factor, (w // factor) * factor
    a = arr[:nh, :nw].reshape(nh // factor, factor, nw // factor, factor)
    m = valid_mask[:nh, :nw].reshape(nh // factor, factor, nw // factor, factor)
    s = (a * m).sum(axis=(1, 3)); c = m.sum(axis=(1, 3))
    out = np.zeros_like(s, dtype=np.float32); ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out, c


def windowed_stats(x, valid, size):
    """Mean and std of x over a (size x size) patch window, valid patches only."""
    m = valid.astype(np.float32)
    xm = np.where(valid, x, 0.0).astype(np.float32)
    cnt = uniform_filter(m, size=size, mode='constant')
    s1 = uniform_filter(xm, size=size, mode='constant')
    s2 = uniform_filter(xm * xm, size=size, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt > 0, s1 / cnt, 0.0)
        var = np.where(cnt > 0, s2 / cnt - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)


def extract(path, patch_size_cm, min_valid_frac):
    """Return the 15-feature patch grid + validity + geo-referencing for one
    orthophoto. Feature order matches rf_v8_model['feature_cols']."""
    with rasterio.open(path) as src:
        gsd_m = abs(src.transform.a)
        factor = max(1, int(round(patch_size_cm / (gsd_m * 100))))
        red = src.read(BAND_RED).astype(np.float32)
        green = src.read(BAND_GREEN).astype(np.float32)
        nir = src.read(BAND_NIR).astype(np.float32)
        rededge = src.read(BAND_REDEDGE).astype(np.float32)
        alpha = (src.read(BAND_ALPHA) if src.count >= BAND_ALPHA
                 else np.full(red.shape, 255, np.uint8))
        crs, tr = src.crs, src.transform

    base_valid = (alpha > 0) & ((red + green + nir + rededge) > 0)
    total = red + green + nir + rededge
    ndvi_full, *_ = compute_indices(red, green, nir, rededge)
    p2 = float(np.percentile(total[base_valid], 2.0))
    valid = base_valid & ~(base_valid & (total < p2) & (ndvi_full < 0.15))

    red_p, cnt = block_mean(red, factor, valid)
    green_p, _ = block_mean(green, factor, valid)
    nir_p, _ = block_mean(nir, factor, valid)
    rededge_p, _ = block_mean(rededge, factor, valid)
    ndvi, ndre, gndvi, gri, sr = compute_indices(red_p, green_p, nir_p, rededge_p)
    patch_valid = cnt >= int(min_valid_frac * factor * factor)

    # 6 neighborhood stats, same definitions/order as the DB builder
    _, ndvi_std3 = windowed_stats(ndvi, patch_valid, 3)
    _, nir_std3 = windowed_stats(nir_p, patch_valid, 3)
    _, gri_std3 = windowed_stats(gri, patch_valid, 3)
    ndvi_mean5, ndvi_std5 = windowed_stats(ndvi, patch_valid, 5)
    nir_mean5, _ = windowed_stats(nir_p, patch_valid, 5)

    feat = np.stack([ndvi, ndre, gndvi, gri, sr, red_p, green_p, nir_p, rededge_p,
                     ndvi_std3, nir_std3, gri_std3, ndvi_mean5, nir_mean5,
                     ndvi_std5], axis=-1).astype(np.float32)
    ph, pw = ndvi.shape
    patch_tr = rasterio.Affine(tr.a * factor, tr.b, tr.c,
                               tr.d, tr.e * factor, tr.f)
    return {'feat': feat, 'valid': patch_valid, 'transform': patch_tr,
            'crs': crs, 'shape': (ph, pw), 'patch_m': factor * gsd_m}


def reproject_to_wgs84(raster, transform, crs, nodata=255):
    ph, pw = raster.shape
    left, bottom, right, top = array_bounds(ph, pw, transform)
    dst_tr, dw, dh = calculate_default_transform(crs, 'EPSG:4326', pw, ph,
                                                 left, bottom, right, top)
    dst = np.full((dh, dw), nodata, np.uint8)
    reproject(source=raster, destination=dst, src_transform=transform,
              src_crs=crs, dst_transform=dst_tr, dst_crs='EPSG:4326',
              src_nodata=nodata, dst_nodata=nodata, resampling=Resampling.nearest)
    return dst, array_bounds(dh, dw, dst_tr)


def labels_to_rgba(raster, lut, nodata=255):
    h, w = raster.shape
    rgba = np.zeros((h, w, 4), np.float32)
    for cid, color in enumerate(lut):
        rgba[raster == cid] = color
    rgba[raster == nodata, 3] = 0.0
    return rgba


def hexof(rgba):
    return '#%02x%02x%02x' % tuple(int(255 * v) for v in rgba[:3])


def training_points(db_path):
    """One (class, lat, lon) per training group (placemark / MRK point)."""
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT class, AVG(lat), AVG(lon) FROM training_patches "
                       "GROUP BY group_id").fetchall()
    con.close()
    return [(c, la, lo) for c, la, lo in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagery', required=True,
                    help='glob for multispectral orthophoto GeoTIFF(s)')
    ap.add_argument('--model', default=os.path.join(HERE, 'rf_v8_model.pkl.gz'))
    ap.add_argument('--db', default=os.path.join(HERE, 'training_data.db'))
    ap.add_argument('--patch-size', type=float, default=30)
    ap.add_argument('--min-valid-frac', type=float, default=0.5)
    ap.add_argument('--opacity', type=float, default=0.8)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'map_output'))
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.imagery))
    if not inputs:
        raise SystemExit(f"No imagery matched: {args.imagery}")
    os.makedirs(args.outdir, exist_ok=True)

    opener = gzip.open if args.model.endswith('.gz') else open
    with opener(args.model, 'rb') as fh:
        bundle = pickle.load(fh)
    clf, classes = bundle['model'], bundle['classes']
    name_to_id = {c: i for i, c in enumerate(classes)}
    lut = [COLOR_BY_NAME[c] for c in classes]
    print(f"Loaded model: {len(classes)} classes {classes}")

    overlays = []
    area = np.zeros(len(classes)); npix = np.zeros(len(classes), np.int64)
    for p in inputs:
        stem = os.path.basename(os.path.dirname(os.path.dirname(p))) or \
            os.path.splitext(os.path.basename(p))[0]
        f = extract(p, args.patch_size, args.min_valid_frac)
        ys, xs = np.nonzero(f['valid'])
        print(f"  {stem}: classifying {len(ys):,} patches...")
        X = f['feat'][ys, xs]
        pred = clf.predict(X)
        ids = np.array([name_to_id[n] for n in pred], np.uint8)

        ph, pw = f['shape']
        raster = np.full((ph, pw), 255, np.uint8); raster[ys, xs] = ids
        pa = f['patch_m'] ** 2
        for c in range(len(classes)):
            k = int((ids == c).sum()); npix[c] += k; area[c] += k * pa / 10_000.0

        tif = os.path.join(args.outdir, f"{stem}_v8.tif")
        with rasterio.open(tif, 'w', driver='GTiff', height=ph, width=pw, count=1,
                           dtype='uint8', crs=f['crs'], transform=f['transform'],
                           nodata=255, compress='lzw') as dst:
            dst.write(raster, 1)
        wgs, (w, s, e, n) = reproject_to_wgs84(raster, f['transform'], f['crs'])
        png = os.path.join(args.outdir, f"{stem}_v8.png")
        plt.imsave(png, labels_to_rgba(wgs, lut))
        overlays.append((stem, png, (w, s, e, n)))

    total = int(npix.sum()); total_ha = float(area.sum())
    print(f"\nClassified {total:,} patches over {total_ha:.2f} ha:")
    for c in range(len(classes)):
        print(f"  {classes[c]:<12} {area[c]:8.2f} ha  {100*npix[c]/max(total,1):5.1f}%")

    wests = [b[0] for _, _, b in overlays]; souths = [b[1] for _, _, b in overlays]
    easts = [b[2] for _, _, b in overlays]; norths = [b[3] for _, _, b in overlays]
    fmap = folium.Map(location=[(min(souths)+max(norths))/2,
                                (min(wests)+max(easts))/2],
                      zoom_start=16, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery', name='Satellite').add_to(fmap)
    folium.TileLayer('OpenStreetMap', name='Street map').add_to(fmap)
    for stem, png, (w, s, e, n) in overlays:
        folium.raster_layers.ImageOverlay(image=png, bounds=[[s, w], [n, e]],
                                          opacity=args.opacity, name=stem,
                                          interactive=False).add_to(fmap)

    tp = training_points(args.db)
    if tp:
        layer = folium.FeatureGroup(name=f'Training points ({len(tp)})', show=True)
        for cls, la, lo in tp:
            folium.CircleMarker([la, lo], radius=4, color='#111', weight=1,
                                fill=True, fill_color=hexof(COLOR_BY_NAME[cls]),
                                fill_opacity=1.0, popup=cls).add_to(layer)
        layer.add_to(fmap)
    fmap.fit_bounds([[min(souths), min(wests)], [max(norths), max(easts)]])

    sw = "".join(
        f"<div style='margin:3px 0'><span style='display:inline-block;width:14px;"
        f"height:14px;background:{hexof(lut[c])};border:1px solid #333;"
        f"vertical-align:middle'></span> <b>{classes[c]}</b> &mdash; "
        f"{area[c]:.2f} ha ({100*npix[c]/max(total,1):.1f}%)</div>"
        for c in range(len(classes)))
    legend = (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;"
        "background:rgba(255,255,255,0.94);padding:12px 14px;border:1px solid #888;"
        "border-radius:6px;max-width:330px;font-family:sans-serif;font-size:13px'>"
        f"<div style='font-weight:bold;margin-bottom:6px'>RF v8 species map "
        f"&mdash; {total_ha:.1f} ha</div>{sw}"
        f"<div style='font-size:10px;color:#666;margin-top:6px'>15-feature Random "
        f"Forest (9 spectral + 6 neighborhood), 5 classes. Dots: training points "
        f"coloured by labelled class.</div></div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    html = os.path.join(args.outdir, 'classification_map.html')
    fmap.save(html)
    print(f"\n  wrote {html}\nDone. Open: {html}")


if __name__ == '__main__':
    main()
