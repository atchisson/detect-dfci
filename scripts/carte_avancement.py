"""Carte d'avancement de la détection par département (image pour le README).

Colore en vert les départements traités (sur MapRoulette, repassés avec le modèle
actuel), en orange ceux dont l'inférence est terminée mais pas encore publiée, et
laisse les autres en gris. Les traités sont lus dans
`inference_rattrapage/recap_rattrapage.txt`, les inférés dans les dossiers
`inference_<insee>_2026` / `_standard` ; les contours sont téléchargés une
fois (GeoJSON simplifié) puis mis en cache dans `data/departements.geojson`.

Usage:
    python scripts/carte_avancement.py --out docs/avancement.png
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon

CONTOURS_URL = ("https://raw.githubusercontent.com/gregoiredavid/france-geojson/"
                "master/departements-version-simplifiee.geojson")
UA = "https://github.com/atchisson/detect-dfci"

# Ligne du récap : "  18 Cher   seuil 0.35 : 85 retenue(s) sur 136 candidate(s)"
RECAP = re.compile(r"^\s*(\w+)\s+.+?seuil\s+([\d.]+)\s*:\s*(\d+) retenue")
COULEUR_TRAITE = "#2e8b57"
COULEUR_INFERE = "#f0a030"
COULEUR_RESTE = "#dcdcdc"
INFERENCE_DIR = re.compile(r"^inference_(\w+)_(2026|standard)$")


def lire_inferes(racine: Path, publies: set[str]) -> set[str]:
    """Départements dont toutes les passes `inference_<insee>_*` sont terminées
    (détections écrites, plus de point de reprise) et qui ne sont pas encore
    sur MapRoulette."""
    passes: dict[str, list[bool]] = {}
    for d in racine.iterdir():
        m = INFERENCE_DIR.match(d.name)
        if m and d.is_dir():
            fini = (d / "detected_only.geojson").exists() and not (d / "checkpoint.json").exists()
            passes.setdefault(m.group(1), []).append(fini)
    return {insee for insee, p in passes.items() if all(p)} - publies


def charger_contours(cache: Path) -> dict:
    if not cache.exists():
        req = urllib.request.Request(CONTOURS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(r.read())
    return json.loads(cache.read_text(encoding="utf-8"))


def lire_recap(path: Path) -> dict[str, int]:
    """insee -> nombre de tâches retenues dans le challenge."""
    out: dict[str, int] = {}
    for ligne in path.read_text(encoding="utf-8").splitlines():
        m = RECAP.match(ligne)
        if m:
            out[m.group(1)] = int(m.group(3))
    return out


def anneaux(geom: dict):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        yield poly[0]


def dessiner(contours: dict, taches: dict[str, int], inferes: set[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for f in contours["features"]:
        code = f["properties"]["code"]
        traite = code in taches
        couleur = (COULEUR_TRAITE if traite else COULEUR_INFERE if code in inferes
                   else COULEUR_RESTE)
        for anneau in anneaux(f["geometry"]):
            ax.add_patch(MplPolygon(anneau, closed=True, facecolor=couleur,
                                    edgecolor="white", linewidth=0.5))
        if traite or code in inferes:
            xs = [x for a in anneaux(f["geometry"]) for x, _ in a]
            ys = [y for a in anneaux(f["geometry"]) for _, y in a]
            ax.text((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                    code, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    # Métropole + Corse, pas l'outre-mer.
    ax.set_xlim(-5.5, 10)
    ax.set_ylim(41.2, 51.3)
    ax.set_aspect(1 / 0.69)  # cos(46°) : proportions correctes à nos latitudes
    ax.axis("off")
    ax.set_title("Détection de citernes DFCI — avancement par département", fontsize=13)
    ax.legend(handles=[
        Patch(color=COULEUR_TRAITE,
              label=f"Traité, sur MapRoulette ({len(taches)} dép.)"),
        Patch(color=COULEUR_INFERE,
              label=f"Inféré, revue MapRoulette à venir ({len(inferes)} dép.)"),
        Patch(color=COULEUR_RESTE,
              label=f"Pas encore traité ({96 - len(taches) - len(inferes)} dép. métropole)"),
    ], loc="lower left", frameon=False, fontsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recap", default="inference_rattrapage/recap_rattrapage.txt")
    ap.add_argument("--contours", default="data/departements.geojson")
    ap.add_argument("--racine", default=".", help="dossier contenant les inference_<insee>_*")
    ap.add_argument("--out", default="docs/avancement.png")
    a = ap.parse_args()
    taches = lire_recap(Path(a.recap))
    if not taches:
        raise SystemExit(f"Aucun département lu dans {a.recap}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inferes = lire_inferes(Path(a.racine), set(taches))
    dessiner(charger_contours(Path(a.contours)), taches, inferes, out)
    print(f"{out} : {len(taches)} sur MapRoulette, {len(inferes)} inféré(s) : {sorted(inferes)}")


if __name__ == "__main__":
    main()
