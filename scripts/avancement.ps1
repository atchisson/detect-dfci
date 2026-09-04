<#
.SYNOPSIS
    Avancement d'un run infer_area en cours, avec un ETA fiable.

.DESCRIPTION
    L'ETA imprimé par infer_area.py est calculé depuis le début du run : il
    reste durablement faussé par la mise en route (première tranche de tuiles,
    chargement du modèle). Ce script recalcule le débit sur les N dernières
    lignes de progression seulement.

.EXAMPLE
    .\scripts\avancement.ps1 -Log D:\runs\inference49.log
    .\scripts\avancement.ps1 -Log $log -Window 40   # moyenne plus lisse
#>
param(
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$Window = 15
)

if (-not (Test-Path $Log)) { throw "Journal introuvable : $Log" }

# Lignes du type "  Inférence: 30016/669245 (4%) — écoulé 54.8 min, ETA ..."
$p = @(Get-Content $Log -Encoding UTF8 | Where-Object { $_ -match '\d+/\d+ \(\d+%\)' })
if ($p.Count -lt 2) { "Pas encore assez de lignes de progression."; return }

$k = [Math]::Min($Window, $p.Count - 1)
$null = $p[-$k - 1] -match '(\d+)/(\d+) \(.*?([\d.]+) min'
$n1 = [int]$Matches[1]; $t1 = [double]$Matches[3]
$null = $p[-1] -match '(\d+)/(\d+) \(.*?([\d.]+) min'
$n2 = [int]$Matches[1]; $total = [int]$Matches[2]; $t2 = [double]$Matches[3]

if ($t2 -le $t1) { "Progression figée sur les $k dernières lignes."; return }
$rate = ($n2 - $n1) / ($t2 - $t1)
$suffixe = ($p[-1] -replace '.*— ', '')

if ($n2 -ge $total) {
    "{0:N0}/{1:N0} — terminé en {2:N1} h — {3}" -f $n2, $total, ($t2 / 60), $suffixe
} else {
    "{0:N0}/{1:N0} ({2:P1}) — {3:N0} fenetres/min — reste {4:N1} h — {5}" -f `
        $n2, $total, ($n2 / $total), $rate, (($total - $n2) / $rate / 60), $suffixe
}
