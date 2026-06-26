"""Chronological train/val/test split plus walk-forward CV folds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd

from aqi.config import (
    EMBARGO_HOURS,
    TEST_END,
    TEST_START,
    TIMEZONE,
    TRAIN_END,
    VAL_END,
    VAL_START,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TIMEZONE)


@dataclass(frozen=True)
class SplitMasks:
    train: pd.Series
    val:   pd.Series
    test:  pd.Series


def chronological_masks(df: pd.DataFrame, embargo_hours: int = EMBARGO_HOURS) -> SplitMasks:
    """Return boolean masks for train/val/test on `df['timestamp']`.

    Applies a 1-week embargo before each split boundary so lag features in the
    val/test slices cannot peek at training data.
    """
    ts = df["timestamp"]
    train_end = _ts(TRAIN_END)
    val_start = _ts(VAL_START)
    val_end = _ts(VAL_END)
    test_start = _ts(TEST_START)
    test_end = _ts(TEST_END)
    embargo = pd.Timedelta(hours=embargo_hours)

    train_mask = ts <= (train_end - embargo)
    val_mask = (ts >= val_start) & (ts <= val_end - embargo)
    test_mask = (ts >= test_start) & (ts <= test_end)
    return SplitMasks(train_mask, val_mask, test_mask)


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    fold_months: int = 3,
    initial_train_months: int = 18,
    embargo_hours: int = EMBARGO_HOURS,
) -> Iterator[Tuple[pd.Series, pd.Series]]:
    """Yield (train_mask, val_mask) pairs over the training window.

    Initial train window of `initial_train_months`; sliding val window of
    `fold_months`. Train expands forward by `fold_months` between folds.
    """
    ts = df["timestamp"]
    start = ts.min()
    train_end_global = _ts(TRAIN_END)
    embargo = pd.Timedelta(hours=embargo_hours)

    cur_train_end = start + pd.DateOffset(months=initial_train_months)
    for _ in range(n_folds):
        val_start = cur_train_end + embargo
        val_end = val_start + pd.DateOffset(months=fold_months)
        if val_end > train_end_global:
            break
        train_mask = ts <= cur_train_end
        val_mask = (ts >= val_start) & (ts < val_end)
        yield train_mask, val_mask
        cur_train_end = val_end
