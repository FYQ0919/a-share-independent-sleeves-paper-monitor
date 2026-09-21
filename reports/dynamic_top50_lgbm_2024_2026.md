# Historical Dynamic Top50 LGBM Backtest

本次回测把全市场可用历史数据先用于每个历史调仓日的 Top50 选取，再执行原有 LGBM 组合策略。它不把 2026 年的当前 Top50 回填到 2024-2025 年。

## 回测口径

- 数据区间：预热 2024-04-24 至 2026-09-18；正式回测 2024-09-21 至 2026-09-18。
- 全市场输入：5516 只可用股票，正式区间 484 个交易日。
- Top50 选择：每 10 个交易日按当日可见的旧策略规则因子重新排序，取前 50；全市场参与选取，模型训练和排名只使用曾经进入历史 Top50 的股票，并在每个日期再限制为当期 Top50。
- LGBM：75% Alpha158-lite + 25% Alpha158+Barra；模型排名与规则排名按 60/40 融合；月度 expanding LambdaRank。
- 执行：收盘生成信号，下一交易日开盘成交，Top5，10 个交易日调仓，最多替换 1 只，单边成本 12bp。

## 结果

| 组合 | 终值 | 总收益 | 年化 | 夏普 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 历史动态 Top50 LGBM | 1.90x | 89.89% | 38.13% | 1.154 | -31.61% |
| 当前冻结 Top50 对照 | 1.37x | 36.96% | 17.17% | 0.908 | -15.75% |

## 严格检查

- `top50_selection_only_uses_asof_rows`: True
- `top50_available_date_not_after_observation`: True
- `selection_dates_on_declared_grid`: True
- `all_training_labels_mature_before_month_refit`: True
- `signal_close_before_next_open_execution`: True
- `overall_dynamic_no_lookahead_checks`: True
- `strict_point_in_time_full_market`: False

## 股票池覆盖

- 选股快照期数：59；Top50 记录数：2650。
- 正式区间每日已加载 Top50 覆盖率：95.87%；每日成员数范围：49 / 50 / 50。
- 成员区间文件：`reports\dynamic_top50_lgbm_2024_2026_membership_intervals.csv`；每期 Top50 文件：`reports\dynamic_top50_lgbm_2024_2026_top50.csv`。

## 限制

- 该结果的 Top50 选择和模型训练顺序已按历史日期隔离，但数据集仍来自 2026-09-18 的当前股票代码快照，未包含历史退市股票。
- 价格是当前版本前复权 qfq，不是严格 as-of 复权价；历史每日 ST、停牌、涨跌停和可交易状态没有完整回溯。
- 因此结果标签是 `dynamic_top50_historical_observed_universe`，不是 `strict_point_in_time_full_market`，不能直接当作无偏生产收益承诺。

## 文件

- JSON：`data\dynamic_top50_lgbm_2024_2026.json`
- 曲线：`reports\dynamic_top50_lgbm_2024_2026_curve.csv`
- 图：`reports\dynamic_top50_lgbm_2024_2026.svg`
