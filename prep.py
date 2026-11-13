"""prep.py - load the data, split into treated/control, measure the overall effect."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Columns that must NEVER be used as inputs (they are results, not causes)
NEVER_USE = ["segment", "treatment", "visit", "conversion", "spend",
             "exposure", "history_segment"]


def make_treatment(series):
    """1 = got the offer, 0 = did not. Works for 0/1 columns and for text like 'No E-Mail'."""
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    return (series != "No E-Mail").astype(int)


def load_data(path_or_file, treat_col="segment", out_col="visit"):
    """Returns X (inputs), t (treated 0/1), y (bought 0/1)."""
    raw = pd.read_csv(path_or_file)
    t = make_treatment(raw[treat_col])
    y = raw[out_col].astype(int)
    X = raw.drop(columns=[c for c in NEVER_USE if c in raw.columns])
    X = pd.get_dummies(X, dtype=int)   # turn text columns into 0/1 columns
    return X, t, y


def overall_effect(t, y):
    """Average treatment effect (ATE) = treated rate minus control rate."""
    yt, yc = y[t == 1], y[t == 0]
    rate_t, rate_c = yt.mean(), yc.mean()
    diff = rate_t - rate_c
    se = np.sqrt(rate_t * (1 - rate_t) / len(yt) + rate_c * (1 - rate_c) / len(yc))
    p = 2 * (1 - stats.norm.cdf(abs(diff / se)))
    return {"treated_rate": rate_t, "control_rate": rate_c, "ate": diff,
            "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            "p_value": p, "n_treated": len(yt), "n_control": len(yc)}


def balance_check(X, t):
    """Randomization check: do treated and control look alike? (SMD > 0.1 = suspicious)"""
    rows = []
    for col in X.columns:
        a, b = X.loc[t == 1, col], X.loc[t == 0, col]
        pooled = np.sqrt((a.var() + b.var()) / 2)
        smd = 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled
        rows.append({"column": col, "smd": round(float(smd), 3), "ok": abs(smd) < 0.1})
    return pd.DataFrame(rows)


def split_data(X, t, y, test_size=0.3, seed=42):
    """Split so that treated/control and buyers/non-buyers are balanced in both parts."""
    strat = t * 2 + y
    return train_test_split(X, t, y, test_size=test_size, random_state=seed, stratify=strat)