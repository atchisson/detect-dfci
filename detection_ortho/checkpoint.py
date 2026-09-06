"""Points de reprise pour l'inférence départementale (pause / reprise).

Un run départemental dure ~19 h : l'interrompre ne doit pas coûter le travail
déjà fait. On enregistre périodiquement le nombre de fenêtres traitées **dans
l'ordre** et les détections accumulées ; la reprise repart de cet index.

La reprise n'est valide que si la liste des fenêtres est **exactement** la même :
l'emprise vient d'OSM et peut avoir été modifiée entre-temps, ce qui décalerait
la grille et rendrait l'index mensonger. D'où l'empreinte, qui couvre les
paramètres du run et la géométrie des fenêtres.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAME = "checkpoint.json"
VERSION = 1


def fingerprint(centers: list[tuple[float, float]], params: dict) -> str:
    """Empreinte du couple (paramètres, liste de fenêtres).

    On hache les paramètres, le nombre de fenêtres et un échantillon de leurs
    coordonnées (début, milieu, fin) : hacher les centaines de milliers de
    centres coûterait cher pour un gain nul.
    """
    h = hashlib.sha256()
    for key in sorted(params):
        h.update(f"{key}={params[key]}\n".encode())
    h.update(f"n={len(centers)}\n".encode())
    if centers:
        idx = {0, len(centers) // 2, len(centers) - 1}
        for i in sorted(idx):
            lon, lat = centers[i]
            h.update(f"{i}:{lon:.7f},{lat:.7f}\n".encode())
    return h.hexdigest()


def save(out_dir, empreinte: str, done: int, detections: list[dict]) -> Path:
    """Écrit le point de reprise de façon atomique (tmp + remplacement).

    L'écriture directe laisserait un JSON tronqué si le processus est tué en
    plein milieu — précisément le cas qu'on cherche à couvrir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / NAME
    tmp = out_dir / (NAME + ".tmp")
    payload = {
        "version": VERSION,
        "fingerprint": empreinte,
        "done": int(done),
        "detections": detections,
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)  # atomique sur Windows comme sur POSIX
    return path


def load(out_dir):
    """Point de reprise lisible, ou None (absent, tronqué, version inconnue)."""
    path = Path(out_dir) / NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != VERSION:
        return None
    if not isinstance(data.get("done"), int) or data["done"] < 0:
        return None
    if not isinstance(data.get("detections"), list):
        return None
    return data


def clear(out_dir) -> None:
    """Supprime le point de reprise et son éventuel temporaire."""
    out_dir = Path(out_dir)
    (out_dir / NAME).unlink(missing_ok=True)
    (out_dir / (NAME + ".tmp")).unlink(missing_ok=True)


def stop_requested(out_dir, name: str = "STOP") -> bool:
    """Vrai si un fichier `STOP` a été déposé dans le dossier de sortie.

    Un run lancé en tâche de fond ne reçoit pas de Ctrl+C : déposer un fichier
    est le seul moyen simple de lui demander de s'arrêter proprement.
    """
    return (Path(out_dir) / name).exists()


def clear_stop(out_dir, name: str = "STOP") -> None:
    """Supprime un éventuel fichier STOP résiduel d'une pause précédente."""
    (Path(out_dir) / name).unlink(missing_ok=True)
