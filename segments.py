"""segments.py - sort people into the 4 classic uplift types."""
import numpy as np
import pandas as pd

ORDER = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]


def assign_segments(p0, p1, tol=0.02):
    """p0 = chance to buy WITHOUT the offer, p1 = chance WITH the offer."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    uplift = p1 - p0
    return np.select(
        [uplift > tol, uplift < -tol, p0 >= np.median(p0)],
        ["Persuadable", "Sleeping Dog", "Sure Thing"],
        default="Lost Cause")


def segment_table(seg, p0, p1):
    df = pd.DataFrame({"segment": seg, "p0": p0, "uplift": np.asarray(p1) - np.asarray(p0)})
    out = df.groupby("segment").agg(people=("p0", "size"), avg_buy_chance=("p0", "mean"),
                                    avg_uplift=("uplift", "mean"))
    out["share"] = out["people"] / out["people"].sum()
    return out.reindex([s for s in ORDER if s in out.index])