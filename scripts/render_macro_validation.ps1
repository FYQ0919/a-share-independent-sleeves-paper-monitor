param(
    [string]$CsvPath = (Join-Path $PSScriptRoot "..\reports\macro_validation_performance.csv"),
    [string]$OutputPath = (Join-Path $PSScriptRoot "..\reports\macro_validation_performance.png"),
    [string]$Title = "宏观覆盖层验证集表现",
    [string]$Subtitle = "2021-01-01 至 2023-12-31 | 严格滞后宏观代理 | 仓位变化另计单边 12bp",
    [string]$AxisTitle = "累计净值（2021-01-01 = 1）",
    [string]$SeriesLabel = "宏观代理策略（含成本）"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$rows = Import-Csv -LiteralPath $CsvPath
if ($rows.Count -lt 2) { throw "Macro validation CSV does not contain enough observations." }

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 1800
$chart.Height = 1050
$chart.BackColor = [System.Drawing.Color]::FromArgb(248, 250, 252)
$chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All
$chart.TextAntiAliasingQuality = [System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality]::High

$area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "macro"
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
$area.AxisY.Title = $AxisTitle
$area.AxisY.TitleFont = New-Object System.Drawing.Font("Microsoft YaHei UI", 13)
$area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11)
$area.AxisY.LabelStyle.Format = "0.00"
$area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(229, 231, 235)
$values = @($rows | ForEach-Object { [double]$_.strategy }) + @($rows | ForEach-Object { [double]$_.macro_strategy }) + @($rows | ForEach-Object { [double]$_.nasdaq })
$minimum = ($values | Measure-Object -Minimum).Minimum
$maximum = ($values | Measure-Object -Maximum).Maximum
$padding = [Math]::Max(($maximum - $minimum) * 0.12, 0.04)
$area.AxisY.Minimum = [Math]::Floor(($minimum - $padding) * 20) / 20
$area.AxisY.Maximum = [Math]::Ceiling(($maximum + $padding) * 20) / 20
$area.AxisY.Interval = 0.10
$chart.ChartAreas.Add($area)

function Add-LineSeries {
    param(
        [string]$Name,
        [string]$Column,
        [System.Drawing.Color]$Color,
        [int]$Width,
        [string]$LabelStyle
    )
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $Name
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
    $series.XValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::DateTime
    $series.YValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Double
    $series.BorderWidth = $Width
    $series.Color = $Color
    $series.ChartArea = "macro"
    foreach ($row in $rows) { [void]$series.Points.AddXY([datetime]$row.date, [double]$row.$Column) }
    $last = $series.Points[$series.Points.Count - 1]
    $last.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle
    $last.MarkerSize = 9
    $last.Label = "{0:N2}" -f $last.YValues[0]
    $last.LabelForeColor = $Color
    $last.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 10, [System.Drawing.FontStyle]::Bold)
    $last.SetCustomProperty("LabelStyle", $LabelStyle)
    $chart.Series.Add($series)
}

Add-LineSeries "基础策略" "strategy" ([System.Drawing.Color]::FromArgb(37, 99, 235)) 4 "Top"
Add-LineSeries $SeriesLabel "macro_strategy" ([System.Drawing.Color]::FromArgb(220, 38, 38)) 4 "Bottom"
Add-LineSeries "纳斯达克综合指数" "nasdaq" ([System.Drawing.Color]::FromArgb(22, 163, 74)) 3 "Right"

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
$legend.Alignment = [System.Drawing.StringAlignment]::Center
$legend.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 13)
$legend.BackColor = [System.Drawing.Color]::Transparent
$chart.Legends.Add($legend)

$titleObject = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$titleObject.Text = $Title
$titleObject.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 21, [System.Drawing.FontStyle]::Bold)
$titleObject.ForeColor = [System.Drawing.Color]::FromArgb(17, 24, 39)
$chart.Titles.Add($titleObject)
$subtitleObject = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$subtitleObject.Text = $Subtitle
$subtitleObject.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11)
$subtitleObject.ForeColor = [System.Drawing.Color]::FromArgb(75, 85, 99)
$chart.Titles.Add($subtitleObject)

$chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()
Write-Output $OutputPath
