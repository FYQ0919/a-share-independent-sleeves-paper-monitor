param(
    [string]$InputCsv = (Join-Path $PSScriptRoot '..\reports\current_best_lgbm_vs_csi300_csi2000_2020_now.csv'),
    [string]$OutputPng = (Join-Path $PSScriptRoot '..\reports\current_best_lgbm_vs_csi300_csi2000_2020_now.png')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$rows = @(Import-Csv -LiteralPath $InputCsv)
if ($rows.Count -lt 2) {
    throw 'The comparison CSV does not contain enough observations.'
}

$series = @(
    [pscustomobject]@{ Key = 'current_best_lgbm'; Label = 'Current best 75/25 LGBM'; Color = [System.Drawing.Color]::FromArgb(23, 107, 135) },
    [pscustomobject]@{ Key = 'csi2000'; Label = 'CSI 2000'; Color = [System.Drawing.Color]::FromArgb(140, 94, 36) },
    [pscustomobject]@{ Key = 'csi300'; Label = 'CSI 300'; Color = [System.Drawing.Color]::FromArgb(194, 65, 59) }
)

$dates = New-Object 'System.Collections.Generic.List[datetime]'
$values = @{}
foreach ($item in $series) {
    $values[$item.Key] = New-Object 'System.Collections.Generic.List[double]'
}

foreach ($row in $rows) {
    $dates.Add([datetime]::ParseExact($row.date, 'yyyy-MM-dd', $culture))
    foreach ($item in $series) {
        $values[$item.Key].Add([double]::Parse($row.($item.Key), $culture))
    }
}

$allValues = foreach ($item in $series) { $values[$item.Key] }
$minimum = [double](($allValues | Measure-Object -Minimum).Minimum)
$maximum = [double](($allValues | Measure-Object -Maximum).Maximum)
$padding = [math]::Max(($maximum - $minimum) * 0.06, 0.05)
$yMinimum = [math]::Max(0.0, $minimum - $padding)
$yMaximum = $maximum + $padding

$width = 1600
$height = 900
$left = 120
$right = 330
$top = 110
$bottom = 105
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$proxyStart = [datetime]'2026-01-05'

function Get-X([int]$Index) {
    return [single]($left + $plotWidth * $Index / [math]::Max($rows.Count - 1, 1))
}

function Get-Y([double]$Value) {
    return [single]($top + $plotHeight * ($yMaximum - $Value) / ($yMaximum - $yMinimum))
}

function Draw-SeriesSegment {
    param(
        [System.Drawing.Graphics]$Graphics,
        [System.Drawing.Pen]$Pen,
        [System.Collections.Generic.List[double]]$SeriesValues,
        [int]$StartIndex,
        [int]$EndIndex
    )

    if ($EndIndex -le $StartIndex) { return }
    $points = New-Object 'System.Drawing.PointF[]' ($EndIndex - $StartIndex + 1)
    for ($index = $StartIndex; $index -le $EndIndex; $index++) {
        $points[$index - $StartIndex] = [System.Drawing.PointF]::new((Get-X $index), (Get-Y $SeriesValues[$index]))
    }
    $Graphics.DrawLines($Pen, $points)
}

$outputDirectory = Split-Path -Parent $OutputPng
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}

$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::FromArgb(250, 250, 247))

$titleFont = [System.Drawing.Font]::new('Segoe UI', 22, [System.Drawing.FontStyle]::Regular)
$bodyFont = [System.Drawing.Font]::new('Segoe UI', 12, [System.Drawing.FontStyle]::Regular)
$smallFont = [System.Drawing.Font]::new('Segoe UI', 10, [System.Drawing.FontStyle]::Regular)
$textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 41, 51))
$mutedBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(82, 96, 109))
$gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(217, 226, 232), 1)
$framePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(173, 184, 194), 1)

try {
    $graphics.DrawString('Current best LGBM vs CSI 2000 vs CSI 300: 2020 to latest', $titleFont, $textBrush, $left, 28)
    $subtitle = 'Common start 2020-01-02 = 1.00 | Price return | CSI 2000 dashed after 2025-12-31 uses ETF proxy'
    $graphics.DrawString($subtitle, $smallFont, $mutedBrush, $left, 70)

    for ($tickIndex = 0; $tickIndex -le 6; $tickIndex++) {
        $tickValue = $yMinimum + ($yMaximum - $yMinimum) * $tickIndex / 6.0
        $tickY = Get-Y $tickValue
        $graphics.DrawLine($gridPen, $left, $tickY, $left + $plotWidth, $tickY)
        $tickLabel = '{0:N1}x' -f $tickValue
        $size = $graphics.MeasureString($tickLabel, $smallFont)
        $graphics.DrawString($tickLabel, $smallFont, $mutedBrush, $left - $size.Width - 12, $tickY - $size.Height / 2)
    }

    for ($year = $dates[0].Year; $year -le $dates[$dates.Count - 1].Year; $year++) {
        $target = [datetime]::new($year, 1, 1)
        $nearestIndex = 0
        $nearestDistance = [double]::PositiveInfinity
        for ($index = 0; $index -lt $dates.Count; $index++) {
            $distance = [math]::Abs(($dates[$index] - $target).TotalDays)
            if ($distance -lt $nearestDistance) {
                $nearestDistance = $distance
                $nearestIndex = $index
            }
        }
        $tickX = Get-X $nearestIndex
        $label = [string]$year
        $size = $graphics.MeasureString($label, $smallFont)
        $graphics.DrawString($label, $smallFont, $mutedBrush, $tickX - $size.Width / 2, $top + $plotHeight + 15)
    }

    $graphics.DrawRectangle($framePen, $left, $top, $plotWidth, $plotHeight)

    foreach ($item in $series) {
        $solidPen = [System.Drawing.Pen]::new($item.Color, 3)
        $solidPen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
        $solidPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
        $solidPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
        try {
            if ($item.Key -eq 'csi2000') {
                $proxyIndex = 0
                while ($proxyIndex -lt $dates.Count -and $dates[$proxyIndex] -lt $proxyStart) { $proxyIndex++ }
                Draw-SeriesSegment $graphics $solidPen $values[$item.Key] 0 ([math]::Max(0, $proxyIndex - 1))
                $proxyPen = [System.Drawing.Pen]::new($item.Color, 3)
                $proxyPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
                try {
                    Draw-SeriesSegment $graphics $proxyPen $values[$item.Key] ([math]::Max(0, $proxyIndex - 1)) ($dates.Count - 1)
                }
                finally {
                    $proxyPen.Dispose()
                }
            }
            else {
                Draw-SeriesSegment $graphics $solidPen $values[$item.Key] 0 ($dates.Count - 1)
            }
        }
        finally {
            $solidPen.Dispose()
        }

        $lastValue = $values[$item.Key][$values[$item.Key].Count - 1]
        $lastY = Get-Y $lastValue
        $markerBrush = [System.Drawing.SolidBrush]::new($item.Color)
        try {
            $graphics.FillEllipse($markerBrush, $left + $plotWidth - 5, $lastY - 5, 10, 10)
            $graphics.DrawString(('{0}  {1:N2}x' -f $item.Label, $lastValue), $bodyFont, $markerBrush, $left + $plotWidth + 18, $lastY - 10)
        }
        finally {
            $markerBrush.Dispose()
        }
    }

    $axisLabel = 'A-share trading date'
    $axisSize = $graphics.MeasureString($axisLabel, $bodyFont)
    $graphics.DrawString($axisLabel, $bodyFont, $mutedBrush, $left + ($plotWidth - $axisSize.Width) / 2, $height - 42)

    $bitmap.Save($OutputPng, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $framePen.Dispose()
    $gridPen.Dispose()
    $mutedBrush.Dispose()
    $textBrush.Dispose()
    $smallFont.Dispose()
    $bodyFont.Dispose()
    $titleFont.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

Get-Item -LiteralPath $OutputPng | Select-Object FullName, Length, LastWriteTime
