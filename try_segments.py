from prep import load_data, split_data
from models import TLearner
from segments import assign_segments, segment_table

X, t, y = load_data("sample_hillstrom_like.csv")
X_tr, X_te, t_tr, t_te, y_tr, y_te = split_data(X, t, y)
tl = TLearner().fit(X_tr, t_tr, y_tr)
p0, p1 = tl.predict_p0(X_te), tl.predict_p1(X_te)
print(segment_table(assign_segments(p0, p1), p0, p1).round(3))