"""Couverture d'une couche WMTS par département (quels départements sont dispo ?).

Sonde, en mémoire, les tuiles de zoom 13 qui touchent chaque département et
compte celles qui ont de l'imagerie (ni 404 ni tuile blanche). Même sonde que
`infer_area --layer`. Rien n'est écrit sur le disque hormis le CSV de sortie.

Usage:
    python scripts/couverture_dept.py --out out/couverture_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapely.geometry import box, shape
from shapely.prepared import prep

from detection_ortho.coverage import PROBE_ZOOM, probe_tiles

LAYER_2026 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"
# Départements déjà traités : écartés par défaut.
TRAITES = {"18", "28", "36", "37", "41", "44", "45", "49"}


def tile_xy(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 2 ** z
    lon = lambda xx: xx / n * 360 - 180
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


def tiles_of(geom) -> list[tuple[int, int]]:
    """Tuiles de sonde dont l'emprise touche le département."""
    minx, miny, maxx, maxy = geom.bounds
    x0, y1 = tile_xy(minx, miny, PROBE_ZOOM)
    x1, y0 = tile_xy(maxx, maxy, PROBE_ZOOM)
    pg = prep(geom)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if pg.intersects(box(*tile_bounds(x, y, PROBE_ZOOM)))]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contours", type=Path, default=Path("data/departements.geojson"))
    ap.add_argument("--layer", default=LAYER_2026)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--all", action="store_true", help="inclure les départements déjà traités")
    ap.add_argument("--out", type=Path, default=Path("out/couverture_2026.csv"))
    args = ap.parse_args()

    feats = json.loads(args.contours.read_text(encoding="utf-8"))["features"]
    feats = [f for f in feats if args.all or f["properties"]["code"] not in TRAITES]
    par_dept = {f["properties"]["code"]: (f["properties"]["nom"], tiles_of(shape(f["geometry"])))
                for f in feats}
    toutes = {t for _, ts in par_dept.values() for t in ts}
    print(f"{len(par_dept)} département(s), {len(toutes)} tuile(s) de sonde à zoom {PROBE_ZOOM}",
          flush=True)
    statut = probe_tiles(sorted(toutes), args.layer, workers=args.workers)

    rows = []
    for code, (nom, ts) in par_dept.items():
        avec = sum(1 for t in ts if statut[t] is True)
        indet = sum(1 for t in ts if statut[t] is None)
        rows.append((code, nom, len(ts), avec, indet, avec / len(ts) if ts else 0.0))
    rows.sort(key=lambda r: -r[5])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "nom", "tuiles", "avec_donnees", "indetermine", "couverture"])
        for r in rows:
            w.writerow([*r[:5], f"{r[5]:.3f}"])
    print(f"{'code':<5}{'département':<28}{'tuiles':>7}{'avec':>7}{'indét.':>7}{'couverture':>12}")
    for code, nom, n, avec, indet, c in rows:
        if c > 0 or indet:
            print(f"{code:<5}{nom:<28}{n:>7}{avec:>7}{indet:>7}{c:>11.0%}")
    print(f"\n{sum(1 for r in rows if r[5] >= 0.95)} dép. couverts à >= 95 %, "
          f"{sum(1 for r in rows if 0 < r[5] < 0.95)} en partie, "
          f"{sum(1 for r in rows if r[5] == 0)} sans 2026. Détail : {args.out}")


if __name__ == "__main__":
    main()
