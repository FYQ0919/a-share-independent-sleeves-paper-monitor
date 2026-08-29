$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing.Common

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'reports\overnight_factor_lgbm_2020_2026_curves.csv'
$output = Join-Path $root 'reports\overnight_factor_lgbm_2020_2026.png'
$rows = @(Import-Csv -LiteralPath $source)
if ($rows.Count -lt 2) { throw 'Curve CSV has insufficient rows' }

$series = @(
    @{ Key = 'previous_lgbm'; Label = 'Previous LGBM'; Color = [System.Drawing.Color]::FromArgb(23, 107, 135) },
    @{ Key = 'lgbm_factor_feature'; Label = 'LGBM + factor feature'; Color = [System.Drawing.Color]::FromArgb(180, 83, 9) },
    @{ Key = 'lgbm_factor_overlay'; Label = 'LGBM + fixed 15% overlay'; Color = [System.Drawing.Color]::FromArgb(47, 133, 90) }
)

$width = 1400
$height = 800
$left = 105
$right = 320
$top = 90
$bottom = 92
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$values = foreach ($row in $rows) {
    foreach ($item in $series) { [double]$row.($item.Key) }
}
$low = [Math]::Min(0.9, ($values | Measure-Object -Minimum).Minimum)
$high = [Math]::Max(1.1, ($values | Measure-Object -Maximum).Maximum)
$padding = [Math]::Max(($high - $low) * 0.06, 0.05)
$low = [Math]::Max(0.0, $low - $padding)
$high += $padding

function Get-X([int]$index) {
    return [single]($left + $plotWidth * $index / [Math]::Max($rows.Count - 1, 1))
}
function Get-Y([double]$value) {
    return [single]($top + $plotHeight * ($high - $value) / [Math]::Max($high - $low, 1e-12))
}

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

$graphics.DrawString('Overnight factor LGBM comparison, 2020-2026', $titleFont, $foregroundBrush, $left, 25)
$graphics.DrawString('Monthly expanding fit | close signal | next-open | Top5 | 10-session rebalance | 12 bps one-way', $bodyFont, $mutedBrush, $left, 58)

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
foreach ($year in 2020..2026) {
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
    $pen = [System.Drawing.Pen]::new($item.Color, 2.6)
    $graphics.DrawLines($pen, $points)
    $pen.Dispose()
    $lastValue = [double]$rows[-1].($item.Key)
    $endpoints += [pscustomobject]@{ Item = $item; Value = $lastValue; ActualY = (Get-Y $lastValue); LabelY = (Get-Y $lastValue) }
}

$ordered = @($endpoints | Sort-Object LabelY)
for ($index = 1; $index -lt $ordered.Count; $index++) {
    $ordered[$index].LabelY = [Math]::Max($ordered[$index].LabelY, $ordered[$index - 1].LabelY + 26)
}
$overflow = $ordered[-1].LabelY - ($top + $plotHeight - 10)
if ($overflow -gt 0) {
    foreach ($endpoint in $ordered) { $endpoint.LabelY -= $overflow }
}
foreach ($endpoint in $ordered) {
    $colorBrush = [System.Drawing.SolidBrush]::new($endpoint.Item.Color)
    $colorPen = [System.Drawing.Pen]::new($endpoint.Item.Color, 1.4)
    $graphics.FillEllipse($colorBrush, $left + $plotWidth - 4, $endpoint.ActualY - 4, 8, 8)
    $graphics.DrawLine($colorPen, $left + $plotWidth + 5, $endpoint.ActualY, $left + $plotWidth + 22, $endpoint.LabelY)
    $label = '{0}  {1:N2}x' -f $endpoint.Item.Label, $endpoint.Value
    $graphics.DrawString($label, $bodyFont, $foregroundBrush, $left + $plotWidth + 29, $endpoint.LabelY - 8)
    $colorBrush.Dispose()
    $colorPen.Dispose()
}

$xTitle = 'Signal date'
$xSize = $graphics.MeasureString($xTitle, $bodyFont)
$graphics.DrawString($xTitle, $bodyFont, $mutedBrush, $left + ($plotWidth - $xSize.Width) / 2, $height - 36)

$bitmap.Save($output, [System.Drawing.Imaging.ImageFormat]::Png)
$gridPen.Dispose()
$foregroundBrush.Dispose()
$mutedBrush.Dispose()
$titleFont.Dispose()
$bodyFont.Dispose()
$smallFont.Dispose()
$graphics.Dispose()
$bitmap.Dispose()
Write-Output $output
