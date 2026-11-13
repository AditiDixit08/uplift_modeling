from prep import load_data, split_data
from models import TLearner, XLearner

X, t, y = load_data("sample_hillstrom_like.csv")
X_tr, X_te, t_tr, t_te, y_tr, y_te = split_data(X, t, y)

tl = TLearner().fit(X_tr, t_tr, y_tr)
xl = XLearner().fit(X_tr, t_tr, y_tr)
print("T-Learner average predicted uplift:", round(tl.predict(X_te).mean(), 3))
print("X-Learner average predicted uplift:", round(xl.predict(X_te).mean(), 3))
print("Highest single prediction:", round(tl.predict(X_te).max(), 3))