import numpy as np
import pandas as pd
from prep import load_data, split_data
from models import baseline_scores, TLearner, XLearner
from evaluate import qini_score, uplift_at_top

X, t, y = load_data("sample_hillstrom_like.csv")
X_tr, X_te, t_tr, t_te, y_tr, y_te = split_data(X, t, y)

scores = {
    "Baseline": baseline_scores(X_tr, y_tr, X_te),
    "T-Learner": TLearner().fit(X_tr, t_tr, y_tr).predict(X_te),
    "X-Learner": XLearner().fit(X_tr, t_tr, y_tr).predict(X_te),
    "Random": np.random.default_rng(0).random(len(X_te)),
}
print(f"{'Model':<10} {'Qini':>8} {'Uplift top 20%':>15}")
for name, s in scores.items():
    q, u = qini_score(s, t_te, y_te), uplift_at_top(s, t_te, y_te)
    print(f"{name:<10} {q:>8.4f} {u:>15.2%}")

# Practice file only: compare with the hidden truth (real data has no truth file)
truth = pd.read_csv("sample_truth.csv").loc[X_te.index, "true_uplift"]
for name in ["Baseline", "T-Learner", "X-Learner"]:
    print(name, "match with TRUE uplift:", round(np.corrcoef(scores[name], truth)[0, 1], 2))