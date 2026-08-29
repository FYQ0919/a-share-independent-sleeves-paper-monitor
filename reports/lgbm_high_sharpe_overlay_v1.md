# High-Sharpe volatility-target LGBM candidate

Historical research candidate only; production and paper trading are unchanged.

| Window | Curve | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|
| Selection 2020-2023 | Base 75/25 LGBM | 13.23% | 28.65% | 0.577 | -32.07% |
| Selection 2020-2023 | Volatility-target LGBM | 11.67% | 23.77% | 0.584 | -23.33% |
| Revealed diagnostic 2024-latest | Base 75/25 LGBM | 78.66% | 37.01% | 1.747 | -18.67% |
| Revealed diagnostic 2024-latest | Volatility-target LGBM | 61.26% | 27.83% | 1.854 | -18.02% |
| Full 2020-latest | Base 75/25 LGBM | 35.76% | 32.27% | 1.106 | -32.07% |
| Full 2020-latest | Volatility-target LGBM | 29.25% | 25.49% | 1.133 | -25.90% |

## Frozen risk rule

- 60-session volatility estimated only from returns through T-1.
- 22% annualized volatility target; long exposure clipped to 65%-100%.
- Residual capital stays in cash; leverage and shorting are disabled.
- 12 bp charged on every aggregate exposure change.

## Limitations

- The 2024-latest diagnostic window has already been revealed and is not a fresh holdout.
- The current Top50 universe is backfilled and has constituent and survivorship bias.
- Current-vintage adjusted prices are not strict point-in-time prices.
- The overlay is evaluated on a stored strategy return stream, not a broker fill replay.
- A new forward paper window is required before production promotion.
