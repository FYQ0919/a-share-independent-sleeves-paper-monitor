param(
    [string]$InputCsv = (Join-Path $PSScriptRoot '..\reports\equal_weight_strategy_blend_2020_2026.csv'),
    [string]$OutputPng = (Join-Path $PSScriptRoot '..\reports\equal_weight_strategy_blend_2020_2026.png'),
    [string]$Title = '2020-2026 两策略 50/50 等资金组合',
    [string]$Subtitle = '初始资金各 50%，子策略独立复利，不额外强制日间再平衡',
    [string]$Summary = 'Terminal 8.12x   |   CAGR 38.77%   |   Sharpe 1.183   |   Max drawdown -31.20%',
    [string]$Key1 = 'equal_weight_blend',
    [string]$Label1 = '50/50 equal-capital blend',
    [string]$Key2 = 'strongest_candidate',
    [string]$Label2 = 'Trend expert + CSI300 hedge 50%',
    [string]$Key3 = 'formal_lgbm',
    [string]$Label3 = 'Previous formal LGBM',
    [string]$Key4 = '',
    [string]$Label4 = '',
    [string]$Footer = '2020-01-02 to 2026-08-25 | Underlying strategy costs included | No extra sleeve-transfer cost'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$rows = @(Import-Csv -LiteralPath $InputCsv)
if ($rows.Count -lt 2) { throw 'The blend CSV does not contain enough observations.' }

$series = @(
    [pscustomobject]@{ Key = $Key1; Label = $Label1; Color = [System.Drawing.Color]::FromArgb(187, 72, 60); Width = 4.0 },
    [pscustomobject]@{ Key = $Key2; Label = $Label2; Color = [System.Drawing.Color]::FromArgb(37, 130, 111); Width = 2.5 },
    [pscustomobject]@{ Key = $Key3; Label = $Label3; Color = [System.Drawing.Color]::FromArgb(51, 103, 165); Width = 2.5 }
)
if ($Key4) {
    $series += [pscustomobject]@{ Key = $Key4; Label = $Label4; Color = [System.Drawing.Color]::FromArgb(126, 87, 158); Width = 2.4 }
}

$dates = New-Object 'System.Collections.Generic.List[datetime]'
$values = @{}
$drawdowns = @{}
foreach ($item in $series) {
    $values[$item.Key] = New-Object 'System.Collections.Generic.List[double]'
    $drawdowns[$item.Key] = New-Object 'System.Collections.Generic.List[double]'
}
foreach ($row in $rows) {
    $dates.Add([datetime]::ParseExact($row.date, 'yyyy-MM-dd', $culture))
    foreach ($item in $series) {
        $values[$item.Key].Add([double]::Parse($row.($item.Key), $culture))
        $drawdowns[$item.Key].Add([double]::Parse($row.("$($item.Key)_drawdown"), $culture) * 100.0)
    }
}

$allValues = foreach ($item in $series) { $values[$item.Key] }
$mainMaximum = [double](($allValues | Measure-Object -Maximum).Maximum) * 1.05
$mainMinimum = 0.8
$allDrawdowns = foreach ($item in $series) { $drawdowns[$item.Key] }
$drawdownMinimum = [math]::Floor([double](($allDrawdowns | Measure-Object -Minimum).Minimum) / 5.0) * 5.0
$drawdownMaximum = 1.0

$width = 1600
$height = 1050
$left = 120
$right = 360
$mainTop = 190
$mainHeight = 470
$drawdownTop = 760
$drawdownHeight = 190
$plotWidth = $width - $left - $right

function Get-X([int]$Index) {
    return [single]($left + $plotWidth * $Index / [math]::Max($rows.Count - 1, 1))
}
function Get-MainY([double]$Value) {
    return [single]($mainTop + $mainHeight * ($mainMaximum - $Value) / ($mainMaximum - $mainMinimum))
}
function Get-DrawdownY([double]$Value) {
    return [single]($drawdownTop + $drawdownHeight * ($drawdownMaximum - $Value) / ($drawdownMaximum - $drawdownMinimum))
}
function Draw-Series {
    param(
        [System.Drawing.Graphics]$Graphics,
        [System.Drawing.Pen]$Pen,
        [System.Collections.Generic.List[double]]$SeriesValues,
        [scriptblock]$MapY
    )
    $points = New-Object 'System.Drawing.PointF[]' $SeriesValues.Count
    for ($index = 0; $index -lt $SeriesValues.Count; $index++) {
        $points[$index] = [System.Drawing.PointF]::new((Get-X $index), (& $MapY $SeriesValues[$index]))
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
$graphics.Clear([System.Drawing.Color]::FromArgb(247, 248, 250))

$titleFont = [System.Drawing.Font]::new('Microsoft YaHei', 23, [System.Drawing.FontStyle]::Regular)
$subtitleFont = [System.Drawing.Font]::new('Microsoft YaHei', 11, [System.Drawing.FontStyle]::Regular)
$bodyFont = [System.Drawing.Font]::new('Segoe UI', 11, [System.Drawing.FontStyle]::Regular)
$smallFont = [System.Drawing.Font]::new('Segoe UI', 9, [System.Drawing.FontStyle]::Regular)
$textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 41, 51))
$mutedBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(82, 96, 109))
$gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(220, 225, 231), 1)
$framePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(165, 175, 185), 1)
$panelBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)

try {
    $graphics.FillRectangle($panelBrush, $left, $mainTop, $plotWidth, $mainHeight)
    $graphics.FillRectangle($panelBrush, $left, $drawdownTop, $plotWidth, $drawdownHeight)
    $graphics.DrawString($Title, $titleFont, $textBrush, $left, 28)
    $graphics.DrawString($Subtitle, $subtitleFont, $mutedBrush, $left, 76)
    $graphics.DrawString($Summary, $bodyFont, $textBrush, $left, 116)
    $graphics.DrawString('Normalized NAV (start = 1.00)', $bodyFont, $mutedBrush, $left, $mainTop - 28)
    $graphics.DrawString('Drawdown (%)', $bodyFont, $mutedBrush, $left, $drawdownTop - 28)

    for ($tickIndex = 0; $tickIndex -le 5; $tickIndex++) {
        $tickValue = $mainMinimum + ($mainMaximum - $mainMinimum) * $tickIndex / 5.0
        $tickY = Get-MainY $tickValue
        $graphics.DrawLine($gridPen, $left, $tickY, $left + $plotWidth, $tickY)
        $label = '{0:N1}x' -f $tickValue
        $size = $graphics.MeasureString($label, $smallFont)
        $graphics.DrawString($label, $smallFont, $mutedBrush, $left - $size.Width - 12, $tickY - $size.Height / 2)
    }
    for ($tickValue = 0; $tickValue -ge $drawdownMinimum; $tickValue -= 10) {
        $tickY = Get-DrawdownY $tickValue
        $graphics.DrawLine($gridPen, $left, $tickY, $left + $plotWidth, $tickY)
        $label = '{0:N0}%' -f $tickValue
        $size = $graphics.MeasureString($label, $smallFont)
        $graphics.DrawString($label, $smallFont, $mutedBrush, $left - $size.Width - 12, $tickY - $size.Height / 2)
    }

    for ($year = $dates[0].Year; $year -le $dates[$dates.Count - 1].Year; $year++) {
        $target = [datetime]::new($year, 1, 1)
        $nearestIndex = 0
        $nearestDistance = [double]::PositiveInfinity
        for ($index = 0; $index -lt $dates.Count; $index++) {
            $distance = [math]::Abs(($dates[$index] - $target).TotalDays)
            if ($distance -lt $nearestDistance) { $nearestDistance = $distance; $nearestIndex = $index }
        }
        $tickX = Get-X $nearestIndex
        $graphics.DrawLine($gridPen, $tickX, $mainTop, $tickX, $mainTop + $mainHeight)
        $graphics.DrawLine($gridPen, $tickX, $drawdownTop, $tickX, $drawdownTop + $drawdownHeight)
        $label = [string]$year
        $size = $graphics.MeasureString($label, $smallFont)
        $graphics.DrawString($label, $smallFont, $mutedBrush, $tickX - $size.Width / 2, $drawdownTop + $drawdownHeight + 14)
    }

    $graphics.DrawRectangle($framePen, $left, $mainTop, $plotWidth, $mainHeight)
    $graphics.DrawRectangle($framePen, $left, $drawdownTop, $plotWidth, $drawdownHeight)
    $legendY = $mainTop + 10
    foreach ($item in $series) {
        $pen = [System.Drawing.Pen]::new($item.Color, [single]$item.Width)
        $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
        try {
            Draw-Series $graphics $pen $values[$item.Key] ${function:Get-MainY}
            Draw-Series $graphics $pen $drawdowns[$item.Key] ${function:Get-DrawdownY}
            $graphics.DrawLine($pen, $left + $plotWidth + 22, $legendY + 8, $left + $plotWidth + 58, $legendY + 8)
            $lastValue = $values[$item.Key][$values[$item.Key].Count - 1]
            $legendText = '{0}  {1:N2}x' -f $item.Label, $lastValue
            $brush = [System.Drawing.SolidBrush]::new($item.Color)
            try { $graphics.DrawString($legendText, $bodyFont, $brush, $left + $plotWidth + 70, $legendY - 1) }
            finally { $brush.Dispose() }
            $legendY += 43
        }
        finally { $pen.Dispose() }
    }
    $graphics.DrawString($Footer, $smallFont, $mutedBrush, $left, $height - 40)
    $bitmap.Save($OutputPng, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $panelBrush.Dispose(); $framePen.Dispose(); $gridPen.Dispose(); $mutedBrush.Dispose(); $textBrush.Dispose()
    $smallFont.Dispose(); $bodyFont.Dispose(); $subtitleFont.Dispose(); $titleFont.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
}

Get-Item -LiteralPath $OutputPng | Select-Object FullName, Length, LastWriteTime
