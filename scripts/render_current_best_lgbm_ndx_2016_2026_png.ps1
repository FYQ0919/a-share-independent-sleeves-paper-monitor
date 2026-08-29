$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing.Common

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'reports\current_best_lgbm_vs_ndx_2016_2026.csv'
$metricsPath = Join-Path $root 'data\current_best_lgbm_vs_ndx_2016_2026.json'
$output = Join-Path $root 'reports\current_best_lgbm_vs_ndx_2016_2026.png'
$rows = @(Import-Csv -LiteralPath $source)
$metrics = Get-Content -Raw -LiteralPath $metricsPath | ConvertFrom-Json
if ($rows.Count -lt 2) { throw 'Curve CSV has insufficient rows' }

$series = @(
    @{ Key = 'current_best_lgbm'; Label = 'Current best 75/25 LGBM'; Color = [System.Drawing.Color]::FromArgb(23, 107, 135) },
    @{ Key = 'nasdaq_100'; Label = 'Nasdaq-100 (NDX)'; Color = [System.Drawing.Color]::FromArgb(194, 65, 59) }
)
$width = 1400
$height = 800
$left = 105
$right = 325
$top = 110
$bottom = 94
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$values = foreach ($row in $rows) { foreach ($item in $series) { [double]$row.($item.Key) } }
$low = ($values | Measure-Object -Minimum).Minimum
$high = ($values | Measure-Object -Maximum).Maximum
$padding = [Math]::Max(($high - $low) * 0.06, 0.05)
$low = [Math]::Max(0.0, $low - $padding)
$high = $high + $padding

function Get-X([int]$index) { [single]($left + $plotWidth * $index / [Math]::Max($rows.Count - 1, 1)) }
function Get-Y([double]$value) { [single]($top + $plotHeight * ($high - $value) / [Math]::Max($high - $low, 1e-12)) }

$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$background = [System.Drawing.Color]::FromArgb(250, 250, 247)
$foreground = [System.Drawing.Color]::FromArgb(31, 41, 51)
$muted = [System.Drawing.Color]::FromArgb(82, 96, 109)
$grid = [System.Drawing.Color]::FromArgb(217, 226, 232)
$graphics.Clear($background)
$titleFont = [System.Drawing.Font]::new('Segoe UI', 18, [System.Drawing.FontStyle]::Bold)
$bodyFont = [System.Drawing.Font]::new('Segoe UI', 10)
$smallFont = [System.Drawing.Font]::new('Segoe UI', 9)
$foregroundBrush = [System.Drawing.SolidBrush]::new($foreground)
$mutedBrush = [System.Drawing.SolidBrush]::new($muted)
$gridPen = [System.Drawing.Pen]::new($grid, 1)

$graphics.DrawString('Current best LGBM vs Nasdaq-100, 2016-2026', $titleFont, $foregroundBrush, $left, 23)
$subtitle = 'Both start at 1.00 | LGBM next-open after 12 bps one-way cost | NDX price return'
$graphics.DrawString($subtitle, $bodyFont, $mutedBrush, $left, 57)
$summary = 'CAGR: LGBM {0:P2} | NDX {1:P2}    Max drawdown: LGBM {2:P2} | NDX {3:P2}' -f $metrics.strategy.metrics.annual_return, $metrics.nasdaq_100.metrics.annual_return, $metrics.strategy.metrics.max_drawdown, $metrics.nasdaq_100.metrics.max_drawdown
$graphics.DrawString($summary, $bodyFont, $mutedBrush, $left, 77)

for ($tick = 0; $tick -lt 6; $tick++) {
    $value = $low + ($high - $low) * $tick / 5
    $y = Get-Y $value
    $graphics.DrawLine($gridPen, $left, $y, $left + $plotWidth, $y)
    $label = '{0:N1}x' -f $value
    $size = $graphics.MeasureString($label, $smallFont)
    $graphics.DrawString($label, $smallFont, $mutedBrush, $left - $size.Width - 10, $y - 7)
}
$graphics.DrawRectangle($gridPen, $left, $top, $plotWidth, $plotHeight)

$dateRows = for ($index = 0; $index -lt $rows.Count; $index++) {
    [pscustomobject]@{ Index = $index; Date = [datetime]$rows[$index].date }
}
foreach ($year in 2016..2026) {
    $target = [datetime]::new($year, 1, 1)
    $closest = $dateRows | Sort-Object { [Math]::Abs(($_.Date - $target).Ticks) } | Select-Object -First 1
    $x = Get-X $closest.Index
    $label = [string]$year
    $size = $graphics.MeasureString($label, $smallFont)
    $graphics.DrawString($label, $smallFont, $mutedBrush, $x - $size.Width / 2, $top + $plotHeight + 13)
}

$endpoints = @()
foreach ($item in $series) {
    $points = [System.Drawing.PointF[]]::new($rows.Count)
    for ($index = 0; $index -lt $rows.Count; $index++) {
        $points[$index] = [System.Drawing.PointF]::new((Get-X $index), (Get-Y ([double]$rows[$index].($item.Key))))
    }
    $pen = [System.Drawing.Pen]::new($item.Color, 2.7)
    $graphics.DrawLines($pen, $points)
    $pen.Dispose()
    $lastValue = [double]$rows[-1].($item.Key)
    $endpoints += [pscustomobject]@{ Item = $item; Value = $lastValue; Y = (Get-Y $lastValue) }
}
foreach ($endpoint in $endpoints) {
    $colorBrush = [System.Drawing.SolidBrush]::new($endpoint.Item.Color)
    $colorPen = [System.Drawing.Pen]::new($endpoint.Item.Color, 1.4)
    $graphics.FillEllipse($colorBrush, $left + $plotWidth - 4, $endpoint.Y - 4, 8, 8)
    $graphics.DrawLine($colorPen, $left + $plotWidth + 5, $endpoint.Y, $left + $plotWidth + 22, $endpoint.Y)
    $label = '{0}  {1:N2}x' -f $endpoint.Item.Label, $endpoint.Value
    $graphics.DrawString($label, $bodyFont, $foregroundBrush, $left + $plotWidth + 29, $endpoint.Y - 8)
    $colorBrush.Dispose()
    $colorPen.Dispose()
}

$xTitle = 'Calendar date'
$xSize = $graphics.MeasureString($xTitle, $bodyFont)
$graphics.DrawString($xTitle, $bodyFont, $mutedBrush, $left + ($plotWidth - $xSize.Width) / 2, $height - 37)
$bitmap.Save($output, [System.Drawing.Imaging.ImageFormat]::Png)
$gridPen.Dispose(); $foregroundBrush.Dispose(); $mutedBrush.Dispose()
$titleFont.Dispose(); $bodyFont.Dispose(); $smallFont.Dispose()
$graphics.Dispose(); $bitmap.Dispose()
Write-Output $output
