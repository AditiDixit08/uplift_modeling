"""models.py - the baseline model and two uplift models (T-Learner, X-Learner)."""
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def make_classifier():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                          learning_rate=0.05, random_state=0)


def make_regressor():
    return HistGradientBoostingRegressor(max_depth=3, max_iter=120,
                                         learning_rate=0.05, random_state=0)


# ---------- 1. Baseline: "will they buy?" (ignores the offer completely) ----------
def baseline_scores(X_train, y_train, X_test):
    model = make_classifier().fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ---------- 2. T-Learner: two models, then subtract ----------
class TLearner:
    """One model for treated people, one for control people.
    uplift = P(buy if treated) - P(buy if not treated)."""

    def fit(self, X, t, y):
        treated = (t == 1).values
        self.m1 = make_classifier().fit(X[treated], y[treated])
        self.m0 = make_classifier().fit(X[~treated], y[~treated])
        return self

    def predict_p1(self, X):
        return self.m1.predict_proba(X)[:, 1]

    def predict_p0(self, X):
        return self.m0.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.predict_p1(X) - self.predict_p0(X)


# ---------- 3. X-Learner: T-Learner plus a second learning round ----------
class XLearner:
    """Stage 1: same as the T-Learner.
    Stage 2: guess each person's 'effect' and train models to predict that guess.
    Final answer blends the two stage-2 models."""

    def fit(self, X, t, y):
        treated = (t == 1).values
        X1, y1 = X[treated], y[treated]
        X0, y0 = X[~treated], y[~treated]
        self.m1 = make_classifier().fit(X1, y1)
        self.m0 = make_classifier().fit(X0, y0)
        # Effect guesses: what each person did vs what the OTHER model says they would do
        d1 = y1.values - self.m0.predict_proba(X1)[:, 1]
        d0 = self.m1.predict_proba(X0)[:, 1] - y0.values
        self.tau1 = make_regressor().fit(X1, d1)
        self.tau0 = make_regressor().fit(X0, d0)
        self.g = float(treated.mean())   # share of people who were treated
        return self

    def predict(self, X):
        return self.g * self.tau0.predict(X) + (1 - self.g) * self.tau1.predict(X)