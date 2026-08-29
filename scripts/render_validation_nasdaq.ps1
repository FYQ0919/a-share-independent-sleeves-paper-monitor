param(
    [string]$CsvPath = (Join-Path $PSScriptRoot "..\reports\validation_strategy_vs_nasdaq.csv"),
    [string]$OutputPath = (Join-Path $PSScriptRoot "..\reports\validation_strategy_vs_nasdaq.png"),
    [string]$ChartTitle = "验证集净值曲线：优化因子策略 vs 纳斯达克综合指数",
    [string]$ChartSubtitle = "2021-01-01 至 2023-12-31 | 起点归一化为 1 | 纳指为价格指数，不含分红、费用、税收与汇率",
    [string]$BaselineDate = "2020-12-31"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$rows = Import-Csv -LiteralPath $CsvPath
if ($rows.Count -lt 2) {
    throw "Comparison CSV does not contain enough observations."
}

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 1800
$chart.Height = 1050
$chart.BackColor = [System.Drawing.Color]::FromArgb(248, 250, 252)
$chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All
$chart.TextAntiAliasingQuality = [System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality]::High

$area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "validation"
$area.BackColor = [System.Drawing.Color]::White
$area.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(7, 13, 89, 73)
$area.InnerPlotPosition = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(10, 4, 86, 87)
$area.AxisX.Title = "日期"
$area.AxisX.TitleFont = New-Object System.Drawing.Font("Microsoft YaHei UI", 13)
$area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11)
$area.AxisX.LabelStyle.Format = "yyyy-MM"
$area.AxisX.IntervalType = [System.Windows.Forms.DataVisualization.Charting.DateTimeIntervalType]::Months
$area.AxisX.Interval = 6
$area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(229, 231, 235)
$area.AxisX.LineColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
$area.AxisX.MajorTickMark.LineColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
$area.AxisY.Title = "累计净值（$BaselineDate = 1）"
$area.AxisY.TitleFont = New-Object System.Drawing.Font("Microsoft YaHei UI", 13)
$area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11)
$area.AxisY.LabelStyle.Format = "0.00"
$area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(229, 231, 235)
$area.AxisY.LineColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
$area.AxisY.MajorTickMark.LineColor = [System.Drawing.Color]::FromArgb(107, 114, 128)

$values = @($rows | ForEach-Object { [double]$_.strategy }) + @($rows | ForEach-Object { [double]$_.nasdaq })
$minimum = ($values | Measure-Object -Minimum).Minimum
$maximum = ($values | Measure-Object -Maximum).Maximum
$padding = [Math]::Max(($maximum - $minimum) * 0.12, 0.04)
$area.AxisY.Minimum = [Math]::Floor(($minimum - $padding) * 20) / 20
$area.AxisY.Maximum = [Math]::Ceiling(($maximum + $padding) * 20) / 20
$yRange = $area.AxisY.Maximum - $area.AxisY.Minimum
$area.AxisY.Interval = if ($yRange -gt 4) { 0.50 } elseif ($yRange -gt 2) { 0.25 } else { 0.10 }
$chart.ChartAreas.Add($area)

function Add-LineSeries {
    param([string]$Name, [string]$Column, [System.Drawing.Color]$Color)
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $Name
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
    $series.XValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::DateTime
    $series.YValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Double
    $series.BorderWidth = 4
    $series.Color = $Color
    $series.ChartArea = "validation"
    foreach ($row in $rows) {
        [void]$series.Points.AddXY([datetime]$row.date, [double]$row.$Column)
    }
    $last = $series.Points[$series.Points.Count - 1]
    $last.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle
    $last.MarkerSize = 10
    $last.Label = "{0:N2}" -f $last.YValues[0]
    $last.LabelForeColor = $Color
    $last.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11, [System.Drawing.FontStyle]::Bold)
    $chart.Series.Add($series)
}

Add-LineSeries "优化因子策略" "strategy" ([System.Drawing.Color]::FromArgb(37, 99, 235))
Add-LineSeries "纳斯达克综合指数" "nasdaq" ([System.Drawing.Color]::FromArgb(220, 38, 38))

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
$legend.Alignment = [System.Drawing.StringAlignment]::Center
$legend.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 13)
$legend.BackColor = [System.Drawing.Color]::Transparent
$chart.Legends.Add($legend)

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = $ChartTitle
$title.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 21, [System.Drawing.FontStyle]::Bold)
$title.ForeColor = [System.Drawing.Color]::FromArgb(17, 24, 39)
$title.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$chart.Titles.Add($title)

$subtitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$subtitle.Text = $ChartSubtitle
$subtitle.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11)
$subtitle.ForeColor = [System.Drawing.Color]::FromArgb(75, 85, 99)
$subtitle.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$chart.Titles.Add($subtitle)

$outputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
$chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()
Write-Output $OutputPath
