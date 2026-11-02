"""Fusion de la double détection d'un département (ortho 2026 + ortho habituelle).

Lit `inference_<insee>_2026` et `inference_<insee>_standard`, garde les candidats
dont le score atteint le seuil, dédoublonne (10 m) et écrit dans
`inference_<insee>/` : challenge_<insee>.geojson, detected_only.geojson (avec la
couche d'origine `src`), matched.geojson et osm_only.geojson (OSM retrouvé ou non
par au moins une des deux passes). La carte de revue se fait ensuite avec
`make_map.py --dir inference_<insee> --base 2026`.

Usage:
    python scripts/fusion_double.py 61 --seuil 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.geo import dedup_points
from detection_ortho.maproulette import to_maproulette_tasks
from scripts.rattrapage import INSTRUCTION

LAYERS = ("2026", "standard")


def lire(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def ecrire(path: Path, feats: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                               ensure_ascii=False), encoding="utf-8")


def cle(feat: dict) -> tuple:
    return tuple(round(c, 6) for c in feat["geometry"]["coordinates"][:2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("insee")
    ap.add_argument("--seuil", type=float, default=0.5)
    ap.add_argument("--racine", type=Path, default=Path("."))
    a = ap.parse_args()
    racine, insee = a.racine, a.insee

    pts = []
    for src in LAYERS:
        for f in lire(racine / f"inference_{insee}_{src}" / "detected_only.geojson"):
            lon, lat = f["geometry"]["coordinates"][:2]
            score = f["properties"].get("score")
            if score is not None and score >= a.seuil:
                pts.append({"lon": lon, "lat": lat, "score": score, "src": src})
    merged = dedup_points(pts, 10.0)
    par_src = {k: sum(p["src"] == k for p in merged) for k in LAYERS}
    print(f"{insee} : {len(pts)} -> {len(merged)} après dédup (seuil {a.seuil}) {par_src}")

    out = racine / f"inference_{insee}"
    out.mkdir(exist_ok=True)
    fc = to_maproulette_tasks(merged, INSTRUCTION)
    for feat, p in zip(fc["features"], merged):
        feat["properties"]["dept"] = insee
        feat["properties"]["src"] = p["src"]
    (out / f"challenge_{insee}.geojson").write_text(
        json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    ecrire(out / "detected_only.geojson", [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
         "properties": {"score": p["score"], "src": p["src"]}} for p in merged])

    osm = {src: lire(racine / f"inference_{insee}_{src}" / "osm_only.geojson") for src in LAYERS}
    non_retrouves = {cle(f) for f in osm["standard"]}
    ecrire(out / "osm_only.geojson", [f for f in osm["2026"] if cle(f) in non_retrouves])
    matched = {cle(f): f for src in LAYERS
               for f in lire(racine / f"inference_{insee}_{src}" / "matched.geojson")}
    ecrire(out / "matched.geojson", list(matched.values()))
    print(f"  OSM non retrouvé : {sum(cle(f) in non_retrouves for f in osm['2026'])}, "
          f"retrouvé : {len(matched)}")


if __name__ == "__main__":
    main()
