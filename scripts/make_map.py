"""Génère une carte interactive HTML (Leaflet + ortho IGN) de revue des détections.

Usage:
    python scripts/make_map.py --dir inference_out --out inference_out/map.html

Ouvre le .html dans un navigateur. Fond = ortho IGN (celle vue par le modèle).
Panneau de revue : parcourt les CANDIDATS (∉ OSM) un par un, boutons
Vrai / Faux / Passer (+ raccourcis clavier v / f / espace, flèches), progression
sauvegardée (localStorage). Une fois terminé, export CSV de la liste + GeoJSON
des vrais positifs (prêt pour MapRoulette). Aucun réseau à la génération.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (fichier, couleur, libellé, revue?) — la couche "revue" est parcourue une à une.
LAYERS = [
    ("detected_only", "#1e63ff", "Candidat (∉ OSM)", True),
    ("matched", "#1faa4b", "Confirmée (∩ OSM)", False),
    ("osm_only", "#e21c1c", "OSM non détectée", False),
    # Présent seulement PENDANT l'inférence (détections brutes, avant comparaison
    # OSM) : permet de suivre le run en direct. Supprimé à la fin.
    ("detections_live", "#ff8c00", "Détection (live, brute)", False),
]

HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revue des détections citernes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0}
  #panel{position:absolute;top:10px;left:10px;z-index:1000;background:#fff;padding:12px 14px;
    border-radius:8px;font:14px sans-serif;box-shadow:0 1px 6px rgba(0,0,0,.35);width:250px}
  #panel h3{margin:0 0 6px;font-size:15px}
  #prog{color:#555;margin-bottom:8px}
  .btn{display:inline-block;border:0;border-radius:6px;padding:8px 10px;margin:2px 2px;
    font-size:14px;cursor:pointer;color:#fff}
  .vrai{background:#1faa4b}.faux{background:#e21c1c}.skip{background:#888}.nav{background:#444}
  .exp{background:#1e63ff;width:100%;margin-top:6px}
  #counts{margin-top:8px;font-size:13px;color:#333;line-height:1.5}
  #done{color:#1faa4b;font-weight:bold;margin-top:6px;display:none}
  .legend{background:#fff;padding:6px 9px;border-radius:6px;font:12px sans-serif;line-height:1.5;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
  .hint{font-size:11px;color:#888;margin-top:6px}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h3>Revue des candidats</h3>
  <div id="prog">–</div>
  <div id="score">–</div>
  <div>
    <button class="btn vrai" onclick="classify('vrai')">✓ Vrai (v)</button>
    <button class="btn faux" onclick="classify('faux')">✗ Faux (f)</button>
  </div>
  <div>
    <button class="btn nav" onclick="prev()">◀ Préc</button>
    <button class="btn skip" onclick="classify('skip')">Passer (␣)</button>
  </div>
  <div id="counts"></div>
  <div id="done">Revue terminée ✓</div>
  <button class="btn exp" onclick="exportCSV()">⬇ Exporter la liste (CSV)</button>
  <button class="btn exp" onclick="exportTP()">⬇ Exporter les vrais (GeoJSON)</button>
  <div class="hint">Raccourcis : v = vrai, f = faux, espace = passer, ← précédent.</div>
</div>
<script>
var DATA = __DATA__;
var ortho = L.tileLayer(
  "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg",
  {maxNativeZoom:19, maxZoom:21, attribution:"Ortho © IGN / Géoplateforme"});
var osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  {maxZoom:19, attribution:"© OpenStreetMap"});
var map = L.map("map", {layers:[ortho]}).setView([47.39, 0.68], 11);
L.control.layers({"Ortho IGN":ortho, "OpenStreetMap":osm}).addTo(map);

function pkey(lat,lon){ return lat.toFixed(6)+","+lon.toFixed(6); }
var verdicts = JSON.parse(localStorage.getItem("citerne_verdicts")||"{}");
function save(){ localStorage.setItem("citerne_verdicts", JSON.stringify(verdicts)); }

var candidates = [];   // couche de revue
var allMarkers = [];

DATA.layers.forEach(function(layer){
  layer.features.forEach(function(f){
    var c=f.geometry.coordinates, lon=c[0], lat=c[1];
    var score=(f.properties&&f.properties.score!=null)?f.properties.score:null;
    var m=L.circleMarker([lat,lon],{radius:6,color:layer.color,weight:2,fillOpacity:.5});
    m.addTo(map); allMarkers.push(m);
    if(layer.review){
      var rec={lat:lat,lon:lon,score:score,marker:m,key:pkey(lat,lon)};
      var i=candidates.length; candidates.push(rec);
      m.on("click", function(){ goTo(i); });
      styleMarker(rec);
    } else {
      m.bindPopup("<b>"+layer.label+"</b>");
    }
  });
});

function styleMarker(rec){
  var v=verdicts[rec.key];
  var col = v==="vrai"?"#1faa4b" : v==="faux"?"#e21c1c" : v==="skip"?"#888" : "#1e63ff";
  rec.marker.setStyle({color:col, fillColor:col, fillOpacity: v?0.8:0.4, radius: v?7:6, weight:2});
}

var idx = 0;
function firstUnreviewed(){ for(var i=0;i<candidates.length;i++){ if(!verdicts[candidates[i].key]) return i; } return 0; }

function goTo(i){
  if(!candidates.length) return;
  idx = (i+candidates.length)%candidates.length;
  var r=candidates[idx];
  map.setView([r.lat, r.lon], 19);                 // vue d'abord -> marqueurs projetés
  candidates.forEach(function(x){ styleMarker(x); });
  r.marker.setStyle({weight:5, color:"#ffcc00"});   // surbrillance courant
  document.getElementById("prog").innerHTML = "Candidat <b>"+(idx+1)+" / "+candidates.length+"</b>";
  document.getElementById("score").innerHTML = "score : "+(r.score!=null?r.score.toFixed(2):"?")+
     (verdicts[r.key]?(" — <b>"+verdicts[r.key]+"</b>"):"");
  refreshCounts();
}
function classify(v){ if(!candidates.length) return; verdicts[candidates[idx].key]=v; save(); next(); }
function next(){ goTo(idx+1); }
function prev(){ goTo(idx-1); }

function refreshCounts(){
  var nv=0,nf=0,ns=0,nr=0;
  candidates.forEach(function(r){ var v=verdicts[r.key];
    if(v==="vrai")nv++; else if(v==="faux")nf++; else if(v==="skip")ns++; else nr++; });
  document.getElementById("counts").innerHTML =
    "✓ vrais : <b>"+nv+"</b><br>✗ faux : <b>"+nf+"</b><br>⏭ passés : "+ns+"<br>reste : <b>"+nr+"</b>";
  document.getElementById("done").style.display = (nr===0 && candidates.length)?"block":"none";
}

document.addEventListener("keydown", function(e){
  if(e.key==="v") classify("vrai");
  else if(e.key==="f") classify("faux");
  else if(e.key===" "){ e.preventDefault(); classify("skip"); }
  else if(e.key==="ArrowLeft") prev();
  else if(e.key==="ArrowRight") next();
});

function download(name, text, type){
  var b=new Blob([text],{type:type}); var a=document.createElement("a");
  a.href=URL.createObjectURL(b); a.download=name; a.click();
}
function exportCSV(){
  var rows=["index,lat,lon,score,verdict"];
  candidates.forEach(function(r,i){
    rows.push([i+1, r.lat.toFixed(6), r.lon.toFixed(6),
      (r.score!=null?r.score.toFixed(3):""), verdicts[r.key]||"non_revu"].join(","));
  });
  download("verdicts.csv", rows.join("\n"), "text/csv");
}
function exportTP(){
  var feats=candidates.filter(function(r){return verdicts[r.key]==="vrai";})
    .map(function(r){ return {type:"Feature",
      geometry:{type:"Point",coordinates:[r.lon,r.lat]},
      properties:{score:r.score, verdict:"vrai"}}; });
  download("vrais_positifs.geojson",
    JSON.stringify({type:"FeatureCollection",features:feats},null,2),
    "application/geo+json");
}

// légende
var legend=L.control({position:"bottomright"});
legend.onAdd=function(){ var d=L.DomUtil.create("div","legend");
  d.innerHTML="<b>État</b><br>"+
    "<span class='dot' style='background:#1e63ff'></span>non revu<br>"+
    "<span class='dot' style='background:#1faa4b'></span>vrai<br>"+
    "<span class='dot' style='background:#e21c1c'></span>faux<br>"+
    "<span class='dot' style='background:#888'></span>passé";
  return d; };
legend.addTo(map);

if(candidates.length){ idx=firstUnreviewed(); goTo(idx); }
else { map.setView([47.39,0.68],11); document.getElementById("prog").textContent="Aucun candidat."; }
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

    layers = []
    for name, color, label, review in LAYERS:
        path = args.dir / f"{name}.geojson"
        if not path.exists():
            continue
        fc = json.loads(path.read_text(encoding="utf-8"))
        layers.append({"name": name, "color": color, "label": label,
                       "review": review, "features": fc.get("features", [])})

    data = json.dumps({"layers": layers}, ensure_ascii=False)
    out.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    n_review = sum(len(le["features"]) for le in layers if le["review"])
    print(f"Carte de revue écrite : {out} ({n_review} candidat(s) à revoir). "
          f"Ouvrez-la dans un navigateur.")


if __name__ == "__main__":
    main()
