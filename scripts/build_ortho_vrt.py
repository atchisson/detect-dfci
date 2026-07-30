"""Construit un VRT mosaïque depuis un dossier de dalles (sans gdalbuildvrt/osgeo).

Utile quand ni `gdalbuildvrt` (CLI) ni `osgeo.gdal` (bindings Python) ne sont
disponibles : on lit les emprises des dalles avec rasterio et on écrit le XML
VRT directement. Suppose des dalles de MÊME CRS et MÊME résolution, non
recouvrantes (cas BD ORTHO).

Usage:
    python scripts/build_ortho_vrt.py --dir chemin/vers/dalles --out ortho37.vrt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

_GDAL_DTYPE = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
               "float32": "Float32"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True,
                    help="dossier des dalles (recherche récursive)")
    ap.add_argument("--glob", type=str, default="*.jp2",
                    help="motif des dalles (défaut *.jp2)")
    ap.add_argument("--out", type=Path, default=Path("ortho.vrt"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.dir / "**" / args.glob), recursive=True))
    if not files:
        raise SystemExit(f"Aucune dalle {args.glob} sous {args.dir}")
    print(f"{len(files)} dalle(s) trouvée(s).")

    with rasterio.open(files[0]) as d0:
        res = d0.res[0]
        crs_wkt = d0.crs.to_wkt()
        nbands = d0.count
        gdal_dtype = _GDAL_DTYPE.get(d0.dtypes[0], "Byte")

    # Emprise globale + métadonnées par dalle.
    minx = miny = 1e18
    maxx = maxy = -1e18
    dalles = []  # (path, left, top, width, height)
    for f in files:
        with rasterio.open(f) as d:
            b = d.bounds
            dalles.append((f, b.left, b.top, d.width, d.height))
            minx, maxx = min(minx, b.left), max(maxx, b.right)
            miny, maxy = min(miny, b.bottom), max(maxy, b.top)

    W = round((maxx - minx) / res)
    H = round((maxy - miny) / res)

    lines = [
        f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
        f'  <SRS>{escape(crs_wkt)}</SRS>',
        f'  <GeoTransform>{minx}, {res}, 0, {maxy}, 0, {-res}</GeoTransform>',
    ]
    for band in range(1, nbands + 1):
        lines.append(f'  <VRTRasterBand dataType="{gdal_dtype}" band="{band}">')
        for path, left, top, w, h in dalles:
            xoff = round((left - minx) / res)
            yoff = round((maxy - top) / res)
            src = escape(Path(path).resolve().as_posix())
            lines.append(
                f'    <SimpleSource>'
                f'<SourceFilename relativeToVRT="0">{src}</SourceFilename>'
                f'<SourceBand>{band}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
                f'<DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>'
                f'</SimpleSource>')
        lines.append('  </VRTRasterBand>')
    lines.append('</VRTDataset>')

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"VRT écrit : {args.out} ({W}x{H} px, {nbands} bandes). "
          f"À passer à infer_area.py via --ortho {args.out}")


if __name__ == "__main__":
    main()
