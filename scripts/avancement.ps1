<#
.SYNOPSIS
    Avancement d'un run infer_area, en une ligne — ponctuel ou rafraîchi.

.DESCRIPTION
    L'ETA imprimé par infer_area.py est calculé depuis le début du run : il
    reste durablement faussé par la mise en route (première tranche de tuiles,
    chargement du modèle). Ce script recalcule le débit sur les N dernières
    lignes de progression seulement.

    Avec -Watch, la ligne est réécrite sur place toutes les -Interval secondes,
    et la boucle s'arrête d'elle-même quand le run se termine (Ctrl+C sinon).

.EXAMPLE
    .\scripts\avancement.ps1 -Log D:\runs\inference41.log
    .\scripts\avancement.ps1 -Log $log -Watch
    .\scripts\avancement.ps1 -Log $log -Watch -Interval 60 -Window 40
#>
param(
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$Window = 15,
    [switch]$Watch,
    [int]$Interval = 30
)

function Get-Avancement {
    param([string]$Chemin, [int]$Fenetre)

    # Lignes du type "  Inférence: 30016/669245 (4%) — écoulé 54.8 min, ETA ..."
    $p = @(Get-Content $Chemin -Encoding UTF8 | Where-Object { $_ -match '\d+/\d+ \(\d+%\)' })
    if ($p.Count -lt 2) {
        return [pscustomobject]@{ Texte = "Pas encore assez de lignes de progression."; Fini = $false }
    }

    $k = [Math]::Min($Fenetre, $p.Count - 1)
    $null = $p[-$k - 1] -match '(\d+)/(\d+) \(.*?([\d.]+) min'
    $n1 = [int]$Matches[1]; $t1 = [double]$Matches[3]
    $null = $p[-1] -match '(\d+)/(\d+) \(.*?([\d.]+) min'
    $n2 = [int]$Matches[1]; $total = [int]$Matches[2]; $t2 = [double]$Matches[3]
    $suffixe = ($p[-1] -replace '.*— ', '')

    if ($n2 -ge $total) {
        return [pscustomobject]@{
            Texte = "{0:N0}/{1:N0} — terminé en {2:N1} h — {3}" -f $n2, $total, ($t2 / 60), $suffixe
            Fini  = $true
        }
    }
    if ($t2 -le $t1) {
        return [pscustomobject]@{
            Texte = "{0:N0}/{1:N0} ({2:P1}) — progression figée sur les {3} dernières lignes — {4}" -f `
                $n2, $total, ($n2 / $total), $k, $suffixe
            Fini  = $false
        }
    }
    $rate = ($n2 - $n1) / ($t2 - $t1)
    [pscustomobject]@{
        Texte = "{0:N0}/{1:N0} ({2:P1}) — {3:N0} fenetres/min — reste {4:N1} h — {5}" -f `
            $n2, $total, ($n2 / $total), $rate, (($total - $n2) / $rate / 60), $suffixe
        Fini  = $false
    }
}

if (-not $Watch) {
    if (-not (Test-Path $Log)) { throw "Journal introuvable : $Log" }
    (Get-Avancement -Chemin $Log -Fenetre $Window).Texte
    return
}

# --- Mode -Watch : réécriture sur place jusqu'à la fin du run ou Ctrl+C ---
$largeur = 100
try { $largeur = [Math]::Max(40, $Host.UI.RawUI.WindowSize.Width - 1) } catch { }

while ($true) {
    if (-not (Test-Path $Log)) {
        Write-Host ("`r" + "En attente du journal $Log ...".PadRight($largeur)) -NoNewline
    } else {
        $a = Get-Avancement -Chemin $Log -Fenetre $Window
        Write-Host ("`r" + $a.Texte.PadRight($largeur)) -NoNewline
        if ($a.Fini) { Write-Host ""; break }
    }
    Start-Sleep -Seconds $Interval
}
