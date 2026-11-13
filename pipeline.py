"""pipeline.py - runs the whole project in one function so app.py stays short."""
import numpy as np
from prep import load_data, overall_effect, balance_check, split_data
from models import baseline_scores, TLearner, XLearner
from evaluate import qini_curve, qini_score, uplift_at_top
from segments import assign_segments, segment_table


def run_pipeline(source, treat_col="segment", out_col="visit", top=0.20):
    X, t, y = load_data(source, treat_col, out_col)
    X_tr, X_te, t_tr, t_te, y_tr, y_te = split_data(X, t, y)

    tl = TLearner().fit(X_tr, t_tr, y_tr)
    xl = XLearner().fit(X_tr, t_tr, y_tr)
    p0, p1 = tl.predict_p0(X_te), tl.predict_p1(X_te)

    scores = {
        "Baseline (will they buy?)": baseline_scores(X_tr, y_tr, X_te),
        "T-Learner": p1 - p0,
        "X-Learner": xl.predict(X_te),
        "Random": np.random.default_rng(0).random(len(X_te)),
    }

    results, curves = [], {}
    for name, s in scores.items():
        results.append({"model": name,
                        "qini_score": round(qini_score(s, t_te, y_te), 4),
                        "uplift_top_20": round(uplift_at_top(s, t_te, y_te, top), 4)})
        curves[name] = qini_curve(s, t_te, y_te)

    seg = assign_segments(p0, p1)
    return {
        "effect": overall_effect(t, y),
        "balance": balance_check(X, t),
        "results": results,
        "curves": curves,
        "segment_table": segment_table(seg, p0, p1),
        "targets": X_te.assign(segment=seg, buy_chance=p0, uplift=scores["X-Learner"]),
        "n_rows": len(X),
    }