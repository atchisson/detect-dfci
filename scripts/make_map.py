"""Génère une carte interactive HTML (Leaflet + ortho IGN) des détections.

Usage:
    python scripts/make_map.py --dir inference_out --out inference_out/map.html

Ouvre le fichier .html produit dans un navigateur : chaque détection est un point
cliquable sur l'ortho IGN (celle vue par le modèle), coloré par catégorie, avec
des liens de vérification (Géoportail, Google satellite, édition OSM).
Aucun réseau à la génération ; la carte charge les tuiles IGN à l'ouverture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (fichier, couleur, libellé) — l'ordre définit la superposition.
LAYERS = [
    ("detected_only", "#1e63ff", "Candidat (∉ OSM)"),
    ("matched", "#1faa4b", "Confirmée (∩ OSM)"),
    ("osm_only", "#e21c1c", "OSM non détectée"),
]

HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Détections citernes — carte</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;font:13px sans-serif;line-height:1.6;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
  .popup a{display:block;margin-top:3px}
</style>
</head>
<body>
<div id="map"></div>
<script>
var DATA = __DATA__;
var ortho = L.tileLayer(
  "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg",
  {maxNativeZoom:19, maxZoom:21, attribution:"Ortho © IGN / Géoplateforme"});
var osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  {maxZoom:19, attribution:"© OpenStreetMap"});
var map = L.map("map", {layers:[ortho]});
L.control.layers({"Ortho IGN":ortho, "OpenStreetMap":osm}).addTo(map);

var all = [];
DATA.layers.forEach(function(layer){
  layer.features.forEach(function(f){
    var c = f.geometry.coordinates, lon=c[0], lat=c[1];
    var score = (f.properties && f.properties.score!=null) ? f.properties.score.toFixed(2) : "?";
    var m = L.circleMarker([lat,lon], {radius:6, color:layer.color, weight:2, fillOpacity:.5});
    var g = "https://www.google.com/maps/place/"+lat+","+lon+"/@"+lat+","+lon+",150m/data=!3m1!1e3";
    var geo = "https://www.geoportail.gouv.fr/carte?c="+lon+","+lat+"&z=19&l0=ORTHOIMAGERY.ORTHOPHOTOS";
    var edit = "https://www.openstreetmap.org/edit#map=20/"+lat+"/"+lon;
    m.bindPopup("<div class='popup'><b>"+layer.label+"</b><br>score: "+score+
      "<br>"+lat.toFixed(6)+", "+lon.toFixed(6)+
      "<a href='"+geo+"' target='_blank'>▶ Géoportail (ortho IGN)</a>"+
      "<a href='"+g+"' target='_blank'>▶ Google satellite</a>"+
      "<a href='"+edit+"' target='_blank'>▶ Ajouter dans OSM</a></div>");
    m.addTo(map); all.push(m);
  });
});
if(all.length){ map.fitBounds(L.featureGroup(all).getBounds().pad(0.05)); }
else { map.setView([47.39,0.68], 11); }

var legend = L.control({position:"bottomright"});
legend.onAdd = function(){
  var d = L.DomUtil.create("div","legend");
  d.innerHTML = "<b>Détections citernes</b><br>" + DATA.legend.map(function(l){
    return "<span class='dot' style='background:"+l.color+"'></span>"+l.label+" ("+l.n+")";
  }).join("<br>");
  return d;
};
legend.addTo(map);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("inference_out"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.dir / "map.html")

    layers, legend = [], []
    for name, color, label in LAYERS:
        path = args.dir / f"{name}.geojson"
        if not path.exists():
            continue
        fc = json.loads(path.read_text(encoding="utf-8"))
        feats = fc.get("features", [])
        layers.append({"color": color, "label": label, "features": feats})
        legend.append({"color": color, "label": label, "n": len(feats)})

    data = json.dumps({"layers": layers, "legend": legend}, ensure_ascii=False)
    out.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    total = sum(le["n"] for le in legend)
    print(f"Carte écrite : {out} ({total} point(s)). Ouvrez-la dans un navigateur.")


if __name__ == "__main__":
    main()
