$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$baseDir = Split-Path -Parent $PSScriptRoot
$csvPath = Join-Path $baseDir "reports\lgbm_drawdown_optimized_curves.csv"
$jsonPath = Join-Path $baseDir "data\lgbm_drawdown_optimization_v1.json"
$pngPath = Join-Path $baseDir "reports\lgbm_drawdown_optimized.png"
$rows = Import-Csv -LiteralPath $csvPath
$result = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 2160
$chart.Height = 1400
$chart.BackColor = [System.Drawing.Color]::FromArgb(248, 249, 251)

$equityArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "Equity"
$equityArea.BackColor = [System.Drawing.Color]::White
$equityArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(8, 12, 88, 50)
$equityArea.AxisX.Enabled = "False"
$equityArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$equityArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$equityArea.AxisY.LineColor = [System.Drawing.Color]::FromArgb(150, 156, 166)
$equityArea.AxisY.Title = "Normalized net value (start = 1.00)"
$equityArea.AxisY.LabelStyle.Format = "0.0"
$equityArea.AxisY.IsStartedFromZero = $false
$chart.ChartAreas.Add($equityArea)

$drawdownArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "Drawdown"
$drawdownArea.BackColor = [System.Drawing.Color]::White
$drawdownArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(8, 66, 88, 25)
$drawdownArea.AlignWithChartArea = "Equity"
$drawdownArea.AlignmentOrientation = "Vertical"
$drawdownArea.AxisX.LabelStyle.Format = "yyyy"
$drawdownArea.AxisX.IntervalType = "Years"
$drawdownArea.AxisX.Interval = 1
$drawdownArea.AxisX.Title = "Date"
$drawdownArea.AxisY.Title = "Drawdown"
$drawdownArea.AxisY.LabelStyle.Format = "0%"
$drawdownArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$drawdownArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$drawdownArea.AxisX.LineColor = [System.Drawing.Color]::FromArgb(150, 156, 166)
$drawdownArea.AxisY.LineColor = [System.Drawing.Color]::FromArgb(150, 156, 166)
$chart.ChartAreas.Add($drawdownArea)

function New-LineSeries($name, $area, $color, $width) {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $name
    $series.ChartArea = $area
    $series.ChartType = "Line"
    $series.BorderWidth = $width
    $series.Color = $color
    $series.XValueType = "DateTime"
    return $series
}

$baseColor = [System.Drawing.Color]::FromArgb(140, 146, 156)
$controlledColor = [System.Drawing.Color]::FromArgb(25, 125, 96)
$baseEquity = New-LineSeries "Full exposure LGBM" "Equity" $baseColor 4
$controlledEquity = New-LineSeries "60d volatility control" "Equity" $controlledColor 5
$baseDrawdown = New-LineSeries "Full exposure drawdown" "Drawdown" $baseColor 3
$controlledDrawdown = New-LineSeries "Controlled drawdown" "Drawdown" $controlledColor 4

foreach ($row in $rows) {
    $when = [DateTime]::ParseExact($row.date, "yyyy-MM-dd", [Globalization.CultureInfo]::InvariantCulture)
    $x = $when.ToOADate()
    [void]$baseEquity.Points.AddXY($x, [double]$row.lgbm_full_exposure)
    [void]$controlledEquity.Points.AddXY($x, [double]$row.lgbm_drawdown_control)
    [void]$baseDrawdown.Points.AddXY($x, [double]$row.full_exposure_drawdown)
    [void]$controlledDrawdown.Points.AddXY($x, [double]$row.controlled_drawdown)
}
$chart.Series.Add($baseEquity)
$chart.Series.Add($controlledEquity)
$chart.Series.Add($baseDrawdown)
$chart.Series.Add($controlledDrawdown)

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = "Top"
$legend.Alignment = "Center"
$legend.Font = New-Object System.Drawing.Font("Segoe UI", 15)
$legend.BackColor = [System.Drawing.Color]::Transparent
$chart.Legends.Add($legend)

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = "LGBM Top5 drawdown control | 2020-2026"
$title.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 23)
$title.ForeColor = [System.Drawing.Color]::FromArgb(30, 35, 43)
$chart.Titles.Add($title)

$selection = $result.winner.selection
$test = $result.winner.test
$subtitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$subtitle.Text = "60-day realized volatility | 22% target | 65% floor | 12bp exposure-change cost`n2020-2023 MDD $(([double]$result.baseline.selection.max_drawdown).ToString('0.0%')) -> $(([double]$selection.max_drawdown).ToString('0.0%'))    2024-2026 MDD $(([double]$result.baseline.test.max_drawdown).ToString('0.0%')) -> $(([double]$test.max_drawdown).ToString('0.0%'))"
$subtitle.Font = New-Object System.Drawing.Font("Segoe UI", 14)
$subtitle.ForeColor = [System.Drawing.Color]::FromArgb(75, 82, 92)
$chart.Titles.Add($subtitle)

$chart.SaveImage($pngPath, "Png")
$chart.Dispose()
Write-Output $pngPath
