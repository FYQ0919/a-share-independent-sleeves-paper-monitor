from datetime import date

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from app.master_reproduction import (
    LightweightAdam,
    MasterPanel,
    MasterReproductionModel,
    build_market_proxy_features,
    robust_scale_market_features,
)


def test_lightweight_adam_updates_parameters_without_torch_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = LightweightAdam([parameter], lr=0.1)
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert float(parameter) < 1.0
    assert parameter.grad is None


def _market_frame(days: int = 90, stocks: int = 12) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=days)
    for day_index, signal_date in enumerate(dates):
        for stock_index in range(stocks):
            rows.append(
                {
                    "date": signal_date,
                    "code": f"s{stock_index:02d}",
                    "ret_1": 0.001 * ((stock_index % 5) - 2) + day_index * 1e-6,
                    "amount": 1_000_000.0 * (stock_index + 1) * (1 + day_index / 1000),
                    "amount_60": 1_000_000.0 * (stock_index + 1),
                }
            )
    return pd.DataFrame(rows)


def test_market_proxy_has_paper_equivalent_dimension_and_is_causal():
    frame = _market_frame()
    original = build_market_proxy_features(frame)
    cutoff = pd.Timestamp("2020-03-02")
    changed = frame.copy()
    changed.loc[changed["date"] > cutoff, ["ret_1", "amount"]] *= 100.0
    modified = build_market_proxy_features(changed)

    assert original.shape[1] == 63
    pd.testing.assert_frame_equal(
        original.loc[original.index <= cutoff],
        modified.loc[modified.index <= cutoff],
    )


def test_market_scaler_uses_training_period_only():
    market = build_market_proxy_features(_market_frame())
    cutoff = date(2020, 3, 2)
    scaled, audit = robust_scale_market_features(market, cutoff)
    changed = market.copy()
    changed.loc[changed.index >= pd.Timestamp(cutoff)] += 1000.0
    _, changed_audit = robust_scale_market_features(changed, cutoff)

    assert audit == changed_audit
    assert np.isfinite(scaled.to_numpy()).all()


def test_master_model_preserves_stock_axis_and_gate_mass():
    model = MasterReproductionModel(
        feature_dim=16,
        market_dim=9,
        d_model=32,
        temporal_heads=4,
        stock_heads=2,
        dropout=0.0,
    )
    stock = torch.randn(11, 8, 16)
    market = torch.randn(9)
    output = model(stock, market)
    gate = model.gate_weights(market)

    assert output.shape == (11,)
    assert gate.shape == (16,)
    assert float(gate.sum()) == pytest.approx(16.0, rel=1e-5)


def test_panel_batch_stops_at_signal_date_and_requires_mature_labels():
    dates = pd.bdate_range("2020-01-01", periods=12)
    codes = [f"s{index:02d}" for index in range(12)]
    features = np.zeros((12, len(codes), 1), dtype=np.float32)
    for day_index in range(12):
        features[day_index, :, 0] = day_index
    labels = np.ones((12, len(codes)), dtype=np.float32)
    label_end = np.empty((12, len(codes)), dtype="datetime64[ns]")
    for day_index, signal_date in enumerate(dates):
        label_end[day_index, :] = np.datetime64(signal_date + pd.offsets.BDay(2))
    panel = MasterPanel(
        frame=pd.DataFrame(
            {
                "date": np.repeat(dates, len(codes)),
                "code": codes * len(dates),
            }
        ),
        feature_columns=["x"],
        market_columns=["m"],
        dates=dates,
        codes=codes,
        feature_values=features,
        market_values=np.zeros((12, 1), dtype=np.float32),
        labels=labels,
        label_end_dates=label_end,
        eligible=np.ones((12, len(codes)), dtype=bool),
        row_lookup=np.arange(12 * len(codes)).reshape(12, len(codes)),
        lookback=8,
    )

    sequence, _, _, _ = panel.batch(8)
    assert sequence.shape == (12, 8, 1)
    assert sequence[0, :, 0].tolist() == list(range(1, 9))
    assert 9.0 not in sequence
    cutoff = dates[10]
    assert panel.training_date_indices(cutoff)[-1] == 7
