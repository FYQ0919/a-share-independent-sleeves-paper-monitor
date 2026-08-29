$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$baseDir = Split-Path -Parent $PSScriptRoot
$csvPath = Join-Path $baseDir "reports\lgbm_vs_nasdaq_2020.csv"
$jsonPath = Join-Path $baseDir "reports\lgbm_vs_nasdaq_2020.json"
$pngPath = Join-Path $baseDir "reports\lgbm_vs_nasdaq_2020.png"
$rows = Import-Csv -LiteralPath $csvPath
$metrics = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 2160
$chart.Height = 1224
$chart.BackColor = [System.Drawing.Color]::FromArgb(248, 249, 251)

$area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "Main"
$area.BackColor = [System.Drawing.Color]::White
$area.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(8, 13, 88, 76)
$area.AxisX.LabelStyle.Format = "yyyy-MM"
$area.AxisX.IntervalType = "Months"
$area.AxisX.Interval = 2
$area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 231, 236)
$area.AxisX.LineColor = [System.Drawing.Color]::FromArgb(150, 156, 166)
$area.AxisY.LineColor = [System.Drawing.Color]::FromArgb(150, 156, 166)
$area.AxisX.Title = "Date"
$area.AxisY.Title = "Normalized net value (start = 1.00)"
$area.AxisY.LabelStyle.Format = "0.00"
$area.AxisY.IsStartedFromZero = $false
$area.AxisY.Minimum = 0.75
$area.AxisY.Maximum = 1.50
$area.AxisY.Interval = 0.10
$chart.ChartAreas.Add($area)

$lgbm = New-Object System.Windows.Forms.DataVisualization.Charting.Series "LGBM LambdaRank Top5"
$lgbm.ChartType = "Line"
$lgbm.BorderWidth = 5
$lgbm.Color = [System.Drawing.Color]::FromArgb(35, 116, 201)
$lgbm.XValueType = "DateTime"

$nasdaq = New-Object System.Windows.Forms.DataVisualization.Charting.Series "Nasdaq Composite"
$nasdaq.ChartType = "Line"
$nasdaq.BorderWidth = 5
$nasdaq.Color = [System.Drawing.Color]::FromArgb(224, 104, 39)
$nasdaq.XValueType = "DateTime"

foreach ($row in $rows) {
    $when = [DateTime]::ParseExact($row.date, "yyyy-MM-dd", [Globalization.CultureInfo]::InvariantCulture)
    [void]$lgbm.Points.AddXY($when.ToOADate(), [double]$row.lgbm)
    [void]$nasdaq.Points.AddXY($when.ToOADate(), [double]$row.nasdaq)
}
$chart.Series.Add($lgbm)
$chart.Series.Add($nasdaq)

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = "Top"
$legend.Alignment = "Center"
$legend.Font = New-Object System.Drawing.Font("Segoe UI", 15)
$legend.BackColor = [System.Drawing.Color]::Transparent
$chart.Legends.Add($legend)

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = "LGBM Top5 vs Nasdaq Composite | 2020 revealed diagnostic"
$title.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 23)
$title.ForeColor = [System.Drawing.Color]::FromArgb(30, 35, 43)
$chart.Titles.Add($title)

$lgbmReturn = [double]$metrics.lgbm.metrics.total_return
$nasdaqReturn = [double]$metrics.nasdaq.metrics.total_return
$subtitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$subtitle.Text = "LGBM $($lgbmReturn.ToString('+0.0%;-0.0%'))    Nasdaq $($nasdaqReturn.ToString('+0.0%;-0.0%'))    |    LGBM includes 12bp one-way cost"
$subtitle.Font = New-Object System.Drawing.Font("Segoe UI", 14)
$subtitle.ForeColor = [System.Drawing.Color]::FromArgb(75, 82, 92)
$chart.Titles.Add($subtitle)

$chart.SaveImage($pngPath, "Png")
$chart.Dispose()
Write-Output $pngPath
