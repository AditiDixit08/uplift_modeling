"""evaluate.py - how do we know an uplift model is any good?
We sort people by predicted uplift and check what REALLY happened to them."""
import numpy as np


def qini_curve(score, t, y):
    """Qini curve: extra buyers gained as we contact more people (best-scored first)."""
    score, t, y = np.asarray(score), np.asarray(t), np.asarray(y)
    order = np.argsort(-score)
    t, y = t[order], y[order]
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    ratio = np.divide(cum_t, cum_c, out=np.zeros(len(t)), where=cum_c > 0)
    qini = cum_yt - cum_yc * ratio
    x = np.arange(1, len(t) + 1) / len(t)
    return np.r_[0, x], np.r_[0, qini]


def qini_score(score, t, y):
    """Area between the model's curve and the 'random' straight line, per customer."""
    x, q = qini_curve(score, t, y)
    random_line = q[-1] * x
    gap = q - random_line
    area = np.sum((x[1:] - x[:-1]) * (gap[1:] + gap[:-1]) / 2)
    return area / len(t)


def uplift_at_top(score, t, y, top=0.20):
    """Real uplift among the top 20% highest-scored people."""
    score, t, y = np.asarray(score), np.asarray(t), np.asarray(y)
    k = int(len(score) * top)
    idx = np.argsort(-score)[:k]
    t_k, y_k = t[idx], y[idx]
    if t_k.sum() == 0 or (1 - t_k).sum() == 0:
        return float("nan")
    return y_k[t_k == 1].mean() - y_k[t_k == 0].mean()