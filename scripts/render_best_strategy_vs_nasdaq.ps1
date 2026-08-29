param(
    [string]$StrategyCurve = "reports/lgbm_new_factors_v1_curves.csv",
    [string]$NasdaqCurve = "reports/lgbm_vs_nasdaq_2020_2026.csv",
    [string]$OutputPath = "reports/best_strategy_vs_nasdaq_2020_2026.png"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing.Common

$strategyRows = Import-Csv $StrategyCurve | ForEach-Object {
    [pscustomobject]@{
        Date = [datetime]$_.date
        Value = [double]$_.selected_model_blend
    }
}
$nasdaqByDate = @{}
Import-Csv $NasdaqCurve | ForEach-Object {
    $nasdaqByDate[$_.date] = [double]$_.nasdaq
}
$points = @($strategyRows | ForEach-Object {
    $key = $_.Date.ToString("yyyy-MM-dd")
    if ($nasdaqByDate.ContainsKey($key)) {
        [pscustomobject]@{
            Date = $_.Date
            Strategy = $_.Value
            Nasdaq = $nasdaqByDate[$key]
        }
    }
})
if ($points.Count -lt 2) {
    throw "策略和纳指曲线没有足够的共同日期"
}

function Get-MaxDrawdown {
    param([object[]]$Rows, [string]$Property)
    $peak = 0.0
    $maxDrawdown = 0.0
    foreach ($row in $Rows) {
        $value = [double]$row.$Property
        if ($value -gt $peak) { $peak = $value }
        $drawdown = $value / $peak - 1.0
        if ($drawdown -lt $maxDrawdown) { $maxDrawdown = $drawdown }
    }
    return $maxDrawdown
}

$first = $points[0]
$last = $points[-1]
$years = ($last.Date - $first.Date).TotalDays / 365.2425
$strategyCagr = [math]::Pow($last.Strategy / $first.Strategy, 1.0 / $years) - 1.0
$nasdaqCagr = [math]::Pow($last.Nasdaq / $first.Nasdaq, 1.0 / $years) - 1.0
$strategyMdd = Get-MaxDrawdown $points "Strategy"
$nasdaqMdd = Get-MaxDrawdown $points "Nasdaq"

$width = 1600
$height = 900
$left = 120
$right = 90
$top = 200
$bottom = 120
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$allValues = @($points.Strategy) + @($points.Nasdaq)
$rawMin = ($allValues | Measure-Object -Minimum).Minimum
$rawMax = ($allValues | Measure-Object -Maximum).Maximum
$yMin = [math]::Max(0.0, [math]::Floor(($rawMin - 0.15) * 2.0) / 2.0)
$yMax = [math]::Ceiling(($rawMax + 0.2) * 2.0) / 2.0
$dateSpan = ($last.Date - $first.Date).TotalDays

$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::FromArgb(250, 251, 252))

$titleFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 25, [System.Drawing.FontStyle]::Bold)
$subtitleFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 12, [System.Drawing.FontStyle]::Regular)
$legendFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 12, [System.Drawing.FontStyle]::Bold)
$axisFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 10, [System.Drawing.FontStyle]::Regular)
$labelFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 11, [System.Drawing.FontStyle]::Bold)
$footnoteFont = [System.Drawing.Font]::new("Microsoft YaHei UI", 9, [System.Drawing.FontStyle]::Regular)
$textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(28, 36, 48))
$mutedBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(92, 104, 120))
$strategyColor = [System.Drawing.Color]::FromArgb(222, 82, 58)
$nasdaqColor = [System.Drawing.Color]::FromArgb(38, 112, 151)
$strategyBrush = [System.Drawing.SolidBrush]::new($strategyColor)
$nasdaqBrush = [System.Drawing.SolidBrush]::new($nasdaqColor)
$strategyPen = [System.Drawing.Pen]::new($strategyColor, 4.0)
$nasdaqPen = [System.Drawing.Pen]::new($nasdaqColor, 4.0)
$gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(218, 223, 230), 1.0)
$axisPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(116, 127, 142), 1.5)

try {
    $graphics.DrawString("75/25 新因子策略 vs 纳斯达克综合指数", $titleFont, $textBrush, 75, 38)
    $subtitle = "共同起点归一化为 1.00  |  $($first.Date.ToString('yyyy-MM-dd')) 至 $($last.Date.ToString('yyyy-MM-dd'))  |  历史诊断"
    $graphics.DrawString($subtitle, $subtitleFont, $mutedBrush, 78, 91)

    $graphics.FillRectangle($strategyBrush, 80, 132, 30, 5)
    $strategyLegend = "75/25 新因子策略   期末 $($last.Strategy.ToString('0.00'))x   复合年化 $($strategyCagr.ToString('0.0%'))   最大回撤 $($strategyMdd.ToString('0.0%'))"
    $graphics.DrawString($strategyLegend, $legendFont, $textBrush, 120, 122)
    $graphics.FillRectangle($nasdaqBrush, 850, 132, 30, 5)
    $nasdaqLegend = "纳斯达克综合   期末 $($last.Nasdaq.ToString('0.00'))x   复合年化 $($nasdaqCagr.ToString('0.0%'))   最大回撤 $($nasdaqMdd.ToString('0.0%'))"
    $graphics.DrawString($nasdaqLegend, $legendFont, $textBrush, 890, 122)

    $scaleX = {
        param([datetime]$Date)
        return [single]($left + (($Date - $first.Date).TotalDays / $dateSpan) * $plotWidth)
    }
    $scaleY = {
        param([double]$Value)
        return [single]($top + (($yMax - $Value) / ($yMax - $yMin)) * $plotHeight)
    }

    for ($tick = $yMin; $tick -le $yMax + 0.001; $tick += 0.5) {
        $y = & $scaleY $tick
        $graphics.DrawLine($gridPen, $left, $y, $left + $plotWidth, $y)
        $label = $tick.ToString("0.0") + "x"
        $size = $graphics.MeasureString($label, $axisFont)
        $graphics.DrawString($label, $axisFont, $mutedBrush, $left - $size.Width - 14, $y - $size.Height / 2)
    }

    $graphics.DrawRectangle($axisPen, $left, $top, $plotWidth, $plotHeight)
    for ($year = $first.Date.Year; $year -le $last.Date.Year; $year++) {
        $tickDate = if ($year -eq $first.Date.Year) { $first.Date } else { [datetime]::new($year, 1, 1) }
        if ($tickDate -gt $last.Date) { continue }
        $x = & $scaleX $tickDate
        $graphics.DrawLine($gridPen, $x, $top, $x, $top + $plotHeight)
        $label = $year.ToString()
        $size = $graphics.MeasureString($label, $axisFont)
        $graphics.DrawString($label, $axisFont, $mutedBrush, $x - $size.Width / 2, $top + $plotHeight + 14)
    }

    $strategyPoints = [System.Drawing.PointF[]]@($points | ForEach-Object {
        [System.Drawing.PointF]::new((& $scaleX $_.Date), (& $scaleY $_.Strategy))
    })
    $nasdaqPoints = [System.Drawing.PointF[]]@($points | ForEach-Object {
        [System.Drawing.PointF]::new((& $scaleX $_.Date), (& $scaleY $_.Nasdaq))
    })
    $graphics.DrawLines($strategyPen, $strategyPoints)
    $graphics.DrawLines($nasdaqPen, $nasdaqPoints)

    $endX = & $scaleX $last.Date
    $strategyEndY = & $scaleY $last.Strategy
    $nasdaqEndY = & $scaleY $last.Nasdaq
    $graphics.FillEllipse($strategyBrush, $endX - 6, $strategyEndY - 6, 12, 12)
    $graphics.FillEllipse($nasdaqBrush, $endX - 6, $nasdaqEndY - 6, 12, 12)
    $graphics.DrawString("4.25x", $labelFont, $strategyBrush, $endX - 74, $strategyEndY - 34)
    $graphics.DrawString("2.88x", $labelFont, $nasdaqBrush, $endX - 74, $nasdaqEndY + 8)

    $graphics.DrawString("累计净值（倍）", $axisFont, $mutedBrush, $left, $top - 30)
    $footnote = "注：策略为冻结协议下的历史诊断，不代表真实前向样本外收益；纳指采用价格收益，不含股息、费用、税费和汇率影响。"
    $graphics.DrawString($footnote, $footnoteFont, $mutedBrush, 80, 845)

    $outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputFullPath)) | Out-Null
    $bitmap.Save($outputFullPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output $outputFullPath
}
finally {
    $strategyPen.Dispose()
    $nasdaqPen.Dispose()
    $gridPen.Dispose()
    $axisPen.Dispose()
    $strategyBrush.Dispose()
    $nasdaqBrush.Dispose()
    $textBrush.Dispose()
    $mutedBrush.Dispose()
    $titleFont.Dispose()
    $subtitleFont.Dispose()
    $legendFont.Dispose()
    $axisFont.Dispose()
    $labelFont.Dispose()
    $footnoteFont.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}
