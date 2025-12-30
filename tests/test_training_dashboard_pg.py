import numpy as np
import pandas as pd
import warnings

from trading.training_dashboard_pg import prepare_progressive_view


def test_progressive_days_accumulates_prior_days():
    log = pd.DataFrame(
        {
            "ts": [
                "2024-01-01 10:00",
                "2024-01-01 11:00",
                "2024-01-02 10:00",
                "2024-01-02 11:00",
                "2024-01-03 12:00",
            ],
            "price": np.arange(5),
        }
    )

    day_idx = 0
    day_keys = None
    visible_lengths = []
    for expected in (2, 4, 5):
        log_view, ts_dt, day_keys, day_idx = prepare_progressive_view(
            log=log, day_idx=day_idx, day_keys=day_keys, refresh_s=1.0, progressive_days=True
        )
        visible_lengths.append(len(log_view))
        assert len(ts_dt) == len(log_view)
        assert ts_dt.index.equals(log_view.index)

    assert visible_lengths == [2, 4, 5]
    assert day_keys is not None and len(day_keys) == 3
    assert day_idx == len(day_keys) - 1  # Stays on the last day once fully revealed.


def test_ts_parsing_coerces_invalid_entries_and_normalizes_days():
    raw_ts = ["bad-ts", "2024-01-01 12:00", "2024/01/02 01:00", None]
    log = pd.DataFrame({"ts": raw_ts, "price": np.arange(len(raw_ts))})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        log_view, ts_dt, day_keys, day_idx = prepare_progressive_view(
            log=log, day_idx=0, day_keys=None, refresh_s=0, progressive_days=True
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected_keys = np.sort(pd.to_datetime(raw_ts, errors="coerce").dropna().normalize().unique())

    assert np.array_equal(day_keys, expected_keys)
    assert day_idx == 0  # No auto-advance when refresh_s=0.
    assert ts_dt.notna().all()
    assert ts_dt.dt.normalize().unique()[0] == expected_keys[0]
    assert len(log_view) == (ts_dt.shape[0])
