"""Rattrapage multi-départements : un run de plusieurs jours, un seul challenge.

Enchaîne `infer_area.py` sur plusieurs départements (un sous-processus par
département, chacun dans `<out-root>/<insee>` avec ses faux déjà rejetés), puis
fusionne leurs candidats en UN challenge MapRoulette.

Reprise : relancer la MÊME commande. Un département terminé est sauté, un
département en pause reprend à son point de reprise. Une pause (Ctrl+C, ou un
fichier STOP déposé dans le dossier du département en cours) arrête aussi ce
script, sans enchaîner sur le suivant. Un échec n'empêche pas les départements
suivants ; il est listé à la fin.

Usage:
    python scripts/rattrapage.py
    python scripts/rattrapage.py --only 18 28 --dry-run
    python scripts/rattrapage.py --merge-only

Génération de FICHIERS uniquement : aucun upload MapRoulette.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho import checkpoint
from detection_ortho.geo import dedup_points
from detection_ortho.geojson_io import write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")
# Les départements couverts par l'ortho 2026 d'abord : ce qui se termine en
# premier est ce qui apporte l'imagerie fraîche. Le 44 et le 49 n'ont pas de 2026.
DEPARTEMENTS = [
    {"insee": "18", "nom": "Cher", "layer": L26},
    {"insee": "28", "nom": "Eure-et-Loir", "layer": L26},
    {"insee": "36", "nom": "Indre", "layer": L26},
    {"insee": "37", "nom": "Indre-et-Loire", "layer": L26},
    {"insee": "41", "nom": "Loir-et-Cher", "layer": L26},
    {"insee": "45", "nom": "Loiret", "layer": L26},
    {"insee": "44", "nom": "Loire-Atlantique", "layer": None},
    {"insee": "49", "nom": "Maine-et-Loire", "layer": None},
]
INFER_AREA = Path(__file__).resolve().parent / "infer_area.py"
DELIVERABLE = "detected_only.geojson"


def dept_out(root, insee) -> Path:
    return Path(root) / insee


def is_paused(out) -> bool:
    """Un point de reprise lisible existe : le run n'est pas terminé."""
    return checkpoint.load(out) is not None


def is_done(out) -> bool:
    """Livrable présent et plus de point de reprise (infer_area l'efface à la fin)."""
    return (Path(out) / DELIVERABLE).is_file() and not is_paused(out)


def build_command(dep, args) -> list[str]:
    cmd = [sys.executable, str(INFER_AREA),
           "--boundary", dep["nom"], "--insee", dep["insee"],
           "--weights", str(args.weights), "--conf", str(args.conf),
           "--known-false", str(Path(args.verdicts_dir) / f"verdicts_{dep['insee']}.csv"),
           "--device", args.device, "--workers", str(args.workers),
           "--cache-gb", str(args.cache_gb),
           "--out", str(dept_out(args.out_root, dep["insee"]))]
    if dep["layer"]:
        cmd += ["--layer", dep["layer"]]
    return cmd


def preflight(deps, args) -> list[str]:
    """Erreurs détectables avant tout calcul (poids, fichiers de faux)."""
    errors = []
    if not Path(args.weights).is_file():
        errors.append(f"poids introuvables : {args.weights}")
    for dep in deps:
        kf = Path(args.verdicts_dir) / f"verdicts_{dep['insee']}.csv"
        if not kf.is_file():
            errors.append(f"fichier de faux introuvable : {kf}")
    return errors


def run_infer(cmd) -> int:
    """Lance infer_area en direct (sa progression reste visible).

    Sur Ctrl+C, le sous-processus reçoit aussi le signal et enregistre son point
    de reprise : on l'attend (jusqu'à 300 s) au lieu de le tuer tout de suite.
    """
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nInterruption : attente de l'arrêt propre d'infer_area "
              "(enregistrement du point de reprise)...", flush=True)
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def run_all(deps, args, run=run_infer) -> dict:
    """Lance les départements dans l'ordre ; retourne {insee: statut}.

    Statuts : « déjà terminé », « terminé », « pause », « échec (code N) »,
    « non traité » (après une pause). Une pause arrête la boucle ; un échec non.
    """
    status: dict[str, str] = {}
    for i, dep in enumerate(deps):
        out = dept_out(args.out_root, dep["insee"])
        label = f"{dep['insee']} {dep['nom']}"
        if is_done(out):
            status[dep["insee"]] = "déjà terminé"
            print(f"=== Département {label} : déjà terminé, sauté ===", flush=True)
            continue
        couche = "ortho 2026" if dep["layer"] else "ortho habituelle"
        print(f"\n=== Département {label} ({couche}) ===", flush=True)
        t0 = time.time()
        code = run(build_command(dep, args))
        heures = (time.time() - t0) / 3600
        if code == 0 and is_done(out):
            status[dep["insee"]] = "terminé"
            print(f"=== {label} terminé en {heures:.1f} h ===", flush=True)
        elif code == 0 and is_paused(out):
            status[dep["insee"]] = "pause"
            for rest in deps[i + 1:]:
                status[rest["insee"]] = "non traité"
            print(f"=== {label} en pause : on s'arrête ici ===", flush=True)
            break
        else:
            status[dep["insee"]] = f"échec (code {code})"
            print(f"=== {label} : ÉCHEC (code {code}), on passe au suivant ===",
                  file=sys.stderr, flush=True)
    return status


def load_candidates(out) -> list[dict]:
    fc = json.loads((Path(out) / DELIVERABLE).read_text(encoding="utf-8"))
    points = []
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"][:2]
        points.append({"lon": lon, "lat": lat,
                       "score": feat.get("properties", {}).get("score")})
    return points


def merge_challenge(root, deps, min_score_2026, min_score_standard,
                    dedup_m=10.0, instruction=INSTRUCTION) -> dict:
    """Fusionne les candidats des départements terminés en un seul challenge.

    Retourne {"fc", "stats", "manquants", "doublons"}. Le seuil dépend de la
    couche du département ; un candidat sans score est écarté ; les doublons
    entre départements voisins (<= dedup_m) gardent le meilleur score.
    """
    kept_all: list[dict] = []
    stats, manquants = [], []
    for dep in deps:
        out = dept_out(root, dep["insee"])
        if not is_done(out):
            manquants.append(dep["insee"])
            continue
        seuil = min_score_2026 if dep["layer"] else min_score_standard
        cands = load_candidates(out)
        kept = [dict(p, dept=dep["insee"]) for p in cands
                if p["score"] is not None and p["score"] >= seuil]
        stats.append({"insee": dep["insee"], "nom": dep["nom"], "seuil": seuil,
                      "candidats": len(cands), "retenus": len(kept)})
        kept_all += kept
    merged = dedup_points(kept_all, dedup_m)
    fc = to_maproulette_tasks(merged, instruction)
    for feat, p in zip(fc["features"], merged):
        feat["properties"]["dept"] = p["dept"]
    return {"fc": fc, "stats": stats, "manquants": manquants,
            "doublons": len(kept_all) - len(merged)}


def format_recap(res, status=None) -> str:
    lines = ["Récapitulatif du rattrapage", ""]
    for s in res["stats"]:
        lines.append(f"  {s['insee']} {s['nom']:<18} seuil {s['seuil']:.2f} : "
                     f"{s['retenus']} retenue(s) sur {s['candidats']} candidate(s)")
    lines.append("")
    lines.append(f"  Doublons entre départements fusionnés : {res['doublons']}")
    lines.append(f"  Tâches du challenge : {len(res['fc']['features'])}")
    if res["manquants"]:
        lines.append(f"  ⚠ Départements non terminés (absents du challenge) : "
                     f"{', '.join(res['manquants'])}")
    if status:
        echecs = [f"{k} ({v})" for k, v in status.items() if v.startswith("échec")]
        if echecs:
            lines.append(f"  ⚠ Échecs : {', '.join(echecs)}")
    return "\n".join(lines)


def _write_merge(deps, args, status=None) -> None:
    res = merge_challenge(args.out_root, deps, args.min_score_2026,
                          args.min_score_standard, args.dedup_m)
    out_path = Path(args.out_root) / "challenge_rattrapage.geojson"
    write_geojson(res["fc"], out_path)
    recap = format_recap(res, status)
    (Path(args.out_root) / "recap_rattrapage.txt").write_text(recap + "\n", encoding="utf-8")
    print("\n" + recap)
    print(f"\nChallenge écrit dans {out_path}. À importer manuellement dans MapRoulette.")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, default=Path("inference_rattrapage"),
                    help="dossier racine : un sous-dossier par département")
    ap.add_argument("--weights", type=Path, default=Path("models/citernes-yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache-gb", type=float, default=10.0,
                    help="plafond disque du cache de tuiles de chaque run")
    ap.add_argument("--verdicts-dir", type=Path, default=Path("verdicts_maproulette"),
                    help="dossier des verdicts_<insee>.csv (faux déjà rejetés)")
    ap.add_argument("--only", nargs="+", metavar="INSEE", default=None,
                    help="ne traiter que ces départements (ex. 18 28)")
    ap.add_argument("--min-score-2026", type=float, default=0.35,
                    help="seuil du challenge pour les départements sur l'ortho 2026")
    ap.add_argument("--min-score-standard", type=float, default=0.60,
                    help="seuil du challenge pour les départements sur l'ortho habituelle")
    ap.add_argument("--dedup-m", type=float, default=10.0,
                    help="rayon (m) de fusion des doublons entre départements")
    ap.add_argument("--merge-only", action="store_true",
                    help="ne rien lancer : fusionner les départements déjà terminés")
    ap.add_argument("--dry-run", action="store_true",
                    help="afficher les commandes prévues sans rien lancer")
    args = ap.parse_args(argv)

    deps = DEPARTEMENTS
    if args.only:
        connus = {d["insee"] for d in DEPARTEMENTS}
        inconnus = [c for c in args.only if c not in connus]
        if inconnus:
            ap.error(f"département(s) inconnu(s) : {', '.join(inconnus)} "
                     f"(connus : {', '.join(sorted(connus))})")
        deps = [d for d in DEPARTEMENTS if d["insee"] in args.only]

    if args.merge_only:
        _write_merge(deps, args)
        return 0

    if args.dry_run:
        for dep in deps:
            if is_done(dept_out(args.out_root, dep["insee"])):
                print(f"# {dep['insee']} {dep['nom']} : déjà terminé, serait sauté")
                continue
            print(" ".join(f'"{c}"' if " " in c else c for c in build_command(dep, args)))
        return 0

    errors = preflight(deps, args)
    if errors:
        for e in errors:
            print(f"Erreur : {e}", file=sys.stderr)
        return 2

    try:
        # `run=run_infer` est résolu ICI, à l'appel : le défaut de run_all est figé
        # à la définition et ne verrait pas un remplacement par les tests.
        status = run_all(deps, args, run=run_infer)
    except KeyboardInterrupt:
        print("\nArrêt demandé : relancez la même commande pour reprendre.", file=sys.stderr)
        return 130

    print("\n=== Bilan ===")
    for dep in deps:
        print(f"  {dep['insee']} {dep['nom']:<18} {status.get(dep['insee'], 'non traité')}")
    if "pause" in status.values():
        print("\nPause : relancez la même commande pour reprendre. "
              "(Aucune fusion tant que la pause n'est pas terminée ; --merge-only "
              "pour fusionner les départements déjà terminés.)")
        return 0
    _write_merge(deps, args, status)
    return 1 if any(v.startswith("échec") for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
