FROM python:3.12-slim

# Dépendances système requises par opencv-python (libGL.so.1, glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Couche dédiée aux dépendances Python (cache de build).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du projet.
COPY detection_ortho ./detection_ortho
COPY scripts ./scripts
COPY tests ./tests
COPY pyproject.toml .

ENV PYTHONUNBUFFERED=1

# Par défaut : lance la suite de tests (vérifie que l'image est saine).
CMD ["python", "-m", "pytest", "-q"]
