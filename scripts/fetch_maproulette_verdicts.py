"""Récupère les verdicts de revue d'un projet MapRoulette (lecture seule).

Pour chaque challenge « DECI <dept> » du projet, télécharge les tâches via
l'API publique (GET uniquement, aucune clé, aucun envoi) et écrit un CSV
`verdicts_<dept>.csv` au format `index,lat,lon,score,verdict` :
statut 1 (fixed) -> vrai, statut 2 (not an issue) -> faux, autres ignorés.

Usage:
    python scripts/fetch_maproulette_verdicts.py --out verdicts_maproulette
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.maproulette import (
    dept_from_challenge_name, task_to_row, rows_to_csv, STATUS_VERDICT,
)

API = "https://maproulette.org/api/v2"
UA = "https://github.com/atchisson/detect-dfci"
PAGE = 500


def http_get_json(url: str, tries: int = 4, pause: float = 3.0,
                  timeout: int = 120):
    """GET JSON avec nouvelles tentatives ; lève RuntimeError si tout échoue."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  GET retry {i + 1}/{tries} ({exc})", file=sys.stderr)
            time.sleep(pause)
    raise RuntimeError(f"GET {url} : {last}") from last


def iter_tasks(challenge_id: int):
    """Toutes les tâches d'un challenge, par pages de PAGE."""
    page = 0
    while True:
        batch = http_get_json(
            f"{API}/challenge/{challenge_id}/tasks?limit={PAGE}&page={page}")
        if not batch:
            return
        yield from batch
        if len(batch) < PAGE:
            return
        page += 1


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=64996,
                    help="identifiant du projet MapRoulette")
    ap.add_argument("--out", type=Path, default=Path("verdicts_maproulette"))
    args = ap.parse_args()

    challenges = http_get_json(
        f"{API}/project/{args.project}/challenges?limit=200")
    args.out.mkdir(parents=True, exist_ok=True)

    for ch in challenges:
        dept = dept_from_challenge_name(ch.get("name"))
        if dept is None:
            print(f"Challenge « {ch.get('name')} » ignoré (nom != 'DECI <dept>').")
            continue
        statuses: Counter = Counter()
        rows = []
        n_illisibles = 0
        for task in iter_tasks(ch["id"]):
            statuses[task.get("status")] += 1
            row = task_to_row(task)
            if row is not None:
                rows.append(row)
            elif task.get("status") in STATUS_VERDICT:
                n_illisibles += 1  # statut 1/2 mais tâche inexploitable
        # Écriture seulement une fois le challenge entièrement lu : pas de CSV partiel.
        path = args.out / f"verdicts_{dept}.csv"
        path.write_text(rows_to_csv(rows), encoding="utf-8")
        n_vrai = sum(1 for r in rows if r["verdict"] == "vrai")
        n_ignores = sum(n for s, n in statuses.items() if s not in STATUS_VERDICT)
        print(f"Dép. {dept}: {n_vrai} vrai(s), {len(rows) - n_vrai} faux, "
              f"{n_ignores} ignoré(s), {n_illisibles} illisible(s) "
              f"[statuts: {dict(sorted(statuses.items(), key=lambda kv: str(kv[0])))}]"
              f" -> {path}")


if __name__ == "__main__":
    main()
