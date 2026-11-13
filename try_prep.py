from prep import load_data, overall_effect, balance_check, split_data

X, t, y = load_data("sample_hillstrom_like.csv")
print("Input columns:", list(X.columns))

e = overall_effect(t, y)
print(f"Treated visit rate: {e['treated_rate']:.2%}")
print(f"Control visit rate: {e['control_rate']:.2%}")
print(f"Overall effect: {e['ate']:+.2%} (95% range {e['ci_low']:+.2%} to {e['ci_high']:+.2%})")
print("Balanced?", balance_check(X, t)["ok"].all())

X_tr, X_te, t_tr, t_te, y_tr, y_te = split_data(X, t, y)
print("Train rows:", len(X_tr), " Test rows:", len(X_te))