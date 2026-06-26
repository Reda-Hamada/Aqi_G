"""Tests for chronological splits and walk-forward folds."""
from __future__ import annotations

import pandas as pd

from aqi.config import TIMEZONE
from aqi.splits.walk_forward import chronological_masks, walk_forward_folds


def _make_df():
    ts = pd.date_range("2013-03-01", "2017-02-28 23:00", freq="h", tz=TIMEZONE)
    return pd.DataFrame({"timestamp": ts})


def test_chronological_masks_disjoint_and_ordered():
    df = _make_df()
    m = chronological_masks(df)
    # Disjoint
    assert not (m.train & m.val).any()
    assert not (m.train & m.test).any()
    assert not (m.val & m.test).any()
    # Ordered
    assert df.loc[m.train, "timestamp"].max() < df.loc[m.val, "timestamp"].min()
    assert df.loc[m.val, "timestamp"].max() < df.loc[m.test, "timestamp"].min()


def test_walk_forward_train_grows_and_val_is_after_train():
    df = _make_df()
    folds = list(walk_forward_folds(df, n_folds=4))
    assert len(folds) >= 2
    last_train_size = 0
    for train_mask, val_mask in folds:
        # Train must grow or stay the same.
        assert train_mask.sum() >= last_train_size
        last_train_size = train_mask.sum()
        # Val is strictly after train.
        assert df.loc[train_mask, "timestamp"].max() < df.loc[val_mask, "timestamp"].min()
        assert not (train_mask & val_mask).any()
