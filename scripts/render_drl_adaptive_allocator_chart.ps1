param(
    [string]$InputCsv = "reports\drl_adaptive_allocator_v1_curves.csv",
    [string]$OutputPng = "reports\drl_adaptive_allocator_v1_chart.png",
    [string]$Title = "DRL adaptive allocation vs fixed baselines",
    [string]$Subtitle = "Walk-forward adaptive period: 2023-01-01 to 2026-08-25 | net of allocation and hedge-change costs",
    [string]$DrlLabel = "DRL adaptive",
    [string]$FixedLabel = "Fixed 75/25, same execution",
    [string]$PreviousLabel = "Previous 75/25 independent sleeves",
    [string]$FormalLabel = "Formal LGBM"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$inputPath = Join-Path $root $InputCsv
$outputPath = Join-Path $root $OutputPng
$curveRows = @(Import-Csv -LiteralPath $inputPath | Where-Object {
    [datetime]::Parse($_.date) -ge [datetime]"2022-12-30"
})
if ($curveRows.Count -lt 2) {
    throw "Insufficient adaptive-period curve rows"
}

$series = @(
    @{ Key = "drl_adaptive"; Label = $DrlLabel; Color = "#167A72"; Width = 4.0 },
    @{ Key = "fixed_75_25_same_contract"; Label = $FixedLabel; Color = "#D97706"; Width = 3.0 },
    @{ Key = "previous_75_25_independent_sleeves"; Label = $PreviousLabel; Color = "#3568B8"; Width = 2.5 },
    @{ Key = "formal_lgbm"; Label = $FormalLabel; Color = "#7A4FA3"; Width = 2.5 }
)

$dates = @($curveRows | ForEach-Object { [datetime]::Parse($_.date) })
$values = @{}
$drawdowns = @{}
$metrics = @{}
foreach ($definition in $series) {
    $key = $definition.Key
    $first = [double]$curveRows[0].$key
    [double[]]$normalized = @($curveRows | ForEach-Object { [double]$($_.$key) / $first })
    [double[]]$returns = @()
    [double[]]$dd = @()
    $peak = $normalized[0]
    for ($index = 0; $index -lt $normalized.Count; $index++) {
        if ($index -gt 0) {
            $returns += $normalized[$index] / $normalized[$index - 1] - 1.0
        }
        $peak = [Math]::Max($peak, $normalized[$index])
        $dd += $normalized[$index] / $peak - 1.0
    }
    $mean = ($returns | Measure-Object -Average).Average
    $sumSquares = 0.0
    foreach ($value in $returns) {
        $sumSquares += [Math]::Pow($value - $mean, 2)
    }
    $standardDeviation = [Math]::Sqrt($sumSquares / $returns.Count)
    $terminal = $normalized[-1]
    $metrics[$key] = @{
        Terminal = $terminal
        Cagr = [Math]::Pow($terminal, 252.0 / $returns.Count) - 1.0
        Sharpe = if ($standardDeviation -gt 0) { $mean / $standardDeviation * [Math]::Sqrt(252.0) } else { 0.0 }
        MaxDrawdown = ($dd | Measure-Object -Minimum).Minimum
    }
    $values[$key] = $normalized
    $drawdowns[$key] = $dd
}

function Get-ScaledX {
    param([int]$Index, [int]$Count, [float]$Left, [float]$Right)
    return $Left + ($Right - $Left) * $Index / [Math]::Max(1, $Count - 1)
}

function Get-ScaledY {
    param([double]$Value, [double]$Minimum, [double]$Maximum, [float]$Top, [float]$Bottom)
    return $Bottom - ($Bottom - $Top) * ($Value - $Minimum) / [Math]::Max(1e-12, $Maximum - $Minimum)
}

function Draw-Axes {
    param(
        [System.Drawing.Graphics]$Graphics,
        [datetime[]]$Dates,
        [double]$Minimum,
        [double]$Maximum,
        [float]$Top,
        [float]$Bottom,
        [int]$TickCount,
        [scriptblock]$Formatter,
        [System.Drawing.Font]$AxisFont
    )
    $left = 105.0
    $right = 1345.0
    $framePen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#C7CBD1")), 1
    $gridPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#E2E5E9")), 1
    $axisBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#555D67"))
    $Graphics.DrawRectangle($framePen, $left, $Top, $right - $left, $Bottom - $Top)
    for ($tick = 0; $tick -lt $TickCount; $tick++) {
        $value = $Minimum + ($Maximum - $Minimum) * $tick / [Math]::Max(1, $TickCount - 1)
        $y = Get-ScaledY $value $Minimum $Maximum $Top $Bottom
        $Graphics.DrawLine($gridPen, $left, $y, $right, $y)
        $label = & $Formatter $value
        $size = $Graphics.MeasureString($label, $AxisFont)
        $Graphics.DrawString($label, $AxisFont, $axisBrush, $left - $size.Width - 12, $y - $size.Height / 2)
    }
    for ($year = $Dates[0].Year; $year -le $Dates[-1].Year; $year++) {
        $target = [datetime]"$year-01-01"
        if ($target -lt $Dates[0]) { continue }
        $nearest = -1
        for ($index = 0; $index -lt $Dates.Count; $index++) {
            if ($Dates[$index] -ge $target) { $nearest = $index; break }
        }
        if ($nearest -lt 0) { continue }
        $x = Get-ScaledX $nearest $Dates.Count $left $right
        $Graphics.DrawLine($gridPen, $x, $Top, $x, $Bottom)
        $yearLabel = "$year"
        $size = $Graphics.MeasureString($yearLabel, $AxisFont)
        $Graphics.DrawString($yearLabel, $AxisFont, $axisBrush, $x - $size.Width / 2, $Bottom + 7)
    }
    $framePen.Dispose()
    $gridPen.Dispose()
    $axisBrush.Dispose()
}

function Draw-Curve {
    param(
        [System.Drawing.Graphics]$Graphics,
        [double[]]$Data,
        [double]$Minimum,
        [double]$Maximum,
        [float]$Top,
        [float]$Bottom,
        [string]$Color,
        [float]$Width
    )
    $left = 105.0
    $right = 1345.0
    $pen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml($Color)), $Width
    $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    for ($index = 1; $index -lt $Data.Count; $index++) {
        $x1 = Get-ScaledX ($index - 1) $Data.Count $left $right
        $x2 = Get-ScaledX $index $Data.Count $left $right
        $y1 = Get-ScaledY $Data[$index - 1] $Minimum $Maximum $Top $Bottom
        $y2 = Get-ScaledY $Data[$index] $Minimum $Maximum $Top $Bottom
        $Graphics.DrawLine($pen, $x1, $y1, $x2, $y2)
    }
    $pen.Dispose()
}

$bitmap = New-Object System.Drawing.Bitmap 1400, 1100
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$titleFont = New-Object System.Drawing.Font "Segoe UI", 22, ([System.Drawing.FontStyle]::Bold)
$subtitleFont = New-Object System.Drawing.Font "Segoe UI", 11
$panelFont = New-Object System.Drawing.Font "Segoe UI", 12, ([System.Drawing.FontStyle]::Bold)
$axisFont = New-Object System.Drawing.Font "Segoe UI", 9
$tableFont = New-Object System.Drawing.Font "Segoe UI", 9
$textBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#20242A"))
$mutedBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#60666F"))

$graphics.DrawString($Title, $titleFont, $textBrush, 42, 22)
$graphics.DrawString($Subtitle, $subtitleFont, $mutedBrush, 44, 60)

$tableX = 660.0
$headers = @("Strategy", "End", "CAGR", "Sharpe", "Max DD")
$columnX = @(660.0, 1080.0, 1160.0, 1240.0, 1325.0)
for ($index = 0; $index -lt $headers.Count; $index++) {
    $graphics.DrawString($headers[$index], $tableFont, $textBrush, $columnX[$index], 82)
}
$rowY = 103.0
foreach ($definition in $series) {
    $metric = $metrics[$definition.Key]
    $swatchPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml($definition.Color)), 4
    $graphics.DrawLine($swatchPen, $tableX, $rowY + 8, $tableX + 18, $rowY + 8)
    $swatchPen.Dispose()
    $graphics.DrawString($definition.Label, $tableFont, $textBrush, $tableX + 25, $rowY)
    $graphics.DrawString(("{0:N2}x" -f $metric.Terminal), $tableFont, $textBrush, 1080, $rowY)
    $graphics.DrawString(("{0:P1}" -f $metric.Cagr), $tableFont, $textBrush, 1160, $rowY)
    $graphics.DrawString(("{0:N3}" -f $metric.Sharpe), $tableFont, $textBrush, 1240, $rowY)
    $graphics.DrawString(("{0:P1}" -f $metric.MaxDrawdown), $tableFont, $textBrush, 1325, $rowY)
    $rowY += 19
}

$netTop = 205.0
$netBottom = 575.0
$netMinimum = [Math]::Min(0.95, (($values.Values | ForEach-Object { $_ } | Measure-Object -Minimum).Minimum * 0.98))
$netMaximum = ($values.Values | ForEach-Object { $_ } | Measure-Object -Maximum).Maximum * 1.04
$graphics.DrawString("Normalized net value", $panelFont, $textBrush, 105, 181)
Draw-Axes $graphics $dates $netMinimum $netMaximum $netTop $netBottom 6 { param($v) "{0:N1}x" -f $v } $axisFont
foreach ($definition in $series) {
    Draw-Curve $graphics $values[$definition.Key] $netMinimum $netMaximum $netTop $netBottom $definition.Color $definition.Width
}

$ddTop = 650.0
$ddBottom = 855.0
$ddMinimum = [Math]::Min(-0.05, (($drawdowns.Values | ForEach-Object { $_ } | Measure-Object -Minimum).Minimum * 1.08))
$graphics.DrawString("Drawdown", $panelFont, $textBrush, 105, 626)
Draw-Axes $graphics $dates $ddMinimum 0.0 $ddTop $ddBottom 5 { param($v) "{0:P0}" -f $v } $axisFont
foreach ($definition in $series) {
    Draw-Curve $graphics $drawdowns[$definition.Key] $ddMinimum 0.0 $ddTop $ddBottom $definition.Color $definition.Width
}

[double[]]$trendWeights = @($curveRows | ForEach-Object { [double]$_.trend_weight })
[double[]]$hedgeRatios = @($curveRows | ForEach-Object { [double]$_.hedge_ratio })
$weightTop = 925.0
$weightBottom = 1035.0
$graphics.DrawString("Executed allocation: trend weight and maximum CSI300 hedge ratio", $panelFont, $textBrush, 105, 901)
Draw-Axes $graphics $dates 0.0 0.90 $weightTop $weightBottom 4 { param($v) "{0:P0}" -f $v } $axisFont
Draw-Curve $graphics $trendWeights 0.0 0.90 $weightTop $weightBottom "#167A72" 3.0
Draw-Curve $graphics $hedgeRatios 0.0 0.90 $weightTop $weightBottom "#C2413B" 2.5
$trendPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#167A72")), 3
$hedgePen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#C2413B")), 2.5
$graphics.DrawLine($trendPen, 1010, 909, 1040, 909)
$graphics.DrawString("Trend weight", $tableFont, $textBrush, 1048, 901)
$graphics.DrawLine($hedgePen, 1160, 909, 1190, 909)
$graphics.DrawString("Max hedge", $tableFont, $textBrush, 1198, 901)
$trendPen.Dispose()
$hedgePen.Dispose()

$graphics.DrawString("Historical diagnostic only. The period was repeatedly inspected; no production configuration was changed.", $axisFont, $mutedBrush, 105, 1070)

$outputDirectory = Split-Path -Parent $outputPath
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$titleFont.Dispose()
$subtitleFont.Dispose()
$panelFont.Dispose()
$axisFont.Dispose()
$tableFont.Dispose()
$textBrush.Dispose()
$mutedBrush.Dispose()
$graphics.Dispose()
$bitmap.Dispose()

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length, LastWriteTime
