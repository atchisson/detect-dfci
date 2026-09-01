"""Reprojette les dalles BD ORTHO en tuiles EPSG:3857 (grille z19) EN PARALLÈLE,
puis assemble une VRT que `infer_area --ortho` lit vite.

Remplace `build_cog.py --src <VRT>` (qui chargeait tout le département en RAM →
COG noir). Chaque dalle est reprojetée dans son propre fichier par un worker
distinct (aucune écriture concurrente). Reprise possible : les tuiles déjà
produites sont sautées.

Usage:
    python scripts/build_cog_tiles.py --src BDORTHO_.../ --out-dir cog_tiles
    # puis: infer_area.py --ortho cog_tiles/mosaic.vrt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.cog import reproject_dalle, write_vrt


def _task(args):
    """Worker : reprojette une dalle. Retourne (out_path, erreur|None)."""
    dalle, out_path, zoom, compress = args
    try:
        reproject_dalle(dalle, out_path, zoom=zoom, compress=compress)
        return (out_path, None)
    except Exception as exc:  # noqa: BLE001
        return (out_path, str(exc))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="dossier des dalles (recherche récursive)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="dossier de sortie des tuiles 3857 + mosaic.vrt")
    ap.add_argument("--glob", type=str, default="*.jp2")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--compress", choices=["jpeg", "deflate"], default="jpeg")
    ap.add_argument("--workers", type=int,
                    default=min((os.cpu_count() or 4), 6),
                    help="processus parallèles (~4 Go RAM chacun)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.src / "**" / args.glob), recursive=True))
    if not files:
        raise SystemExit(f"Aucune dalle {args.glob} sous {args.src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks, skipped = [], 0
    for f in files:
        out = args.out_dir / (Path(f).stem + "_3857.tif")
        if out.exists():
            skipped += 1  # reprise : déjà fait
            continue
        tasks.append((f, str(out), args.zoom, args.compress))

    print(f"{len(files)} dalle(s) ; {skipped} déjà faite(s) ; "
          f"{len(tasks)} à reprojeter sur {args.workers} worker(s) "
          f"(~4 Go RAM/worker).")

    t0 = time.perf_counter()
    last = t0
    done, fails = 0, []
    if tasks:
        with Pool(args.workers) as pool:
            for out, err in pool.imap_unordered(_task, tasks):
                done += 1
                if err:
                    fails.append((out, err))
                    print(f"  ÉCHEC {Path(out).name}: {err}", file=sys.stderr)
                now = time.perf_counter()
                if now - last >= 20.0 or done == len(tasks):
                    el = now - t0
                    rate = done / el if el > 0 else 0
                    eta = (len(tasks) - done) / rate if rate > 0 else 0
                    print(f"  reproject: {done}/{len(tasks)} — écoulé {el/60:.1f} min, "
                          f"ETA {eta/60:.1f} min", flush=True)
                    last = now

    tiles = sorted(str(p) for p in args.out_dir.glob("*_3857.tif"))
    if not tiles:
        raise SystemExit("Aucune tuile produite — rien à assembler.")
    vrt = write_vrt(tiles, args.out_dir / "mosaic.vrt")
    print(f"\n{len(tiles)} tuile(s) ; {len(fails)} échec(s).")
    print(f"VRT écrite : {vrt}")
    print(f"Utilisez : infer_area.py --ortho {vrt}")


if __name__ == "__main__":
    main()
