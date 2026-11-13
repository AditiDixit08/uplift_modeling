"""Creates sample_hillstrom_like.csv: a practice email-campaign dataset.
Same columns as the real Hillstrom dataset, but with the 'truth' planted on purpose."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 64000

recency = rng.integers(1, 13, n)                      # months since last purchase
history = np.round(rng.gamma(2.0, 120, n) + 20, 2)    # dollars spent in the past year
mens = rng.integers(0, 2, n)
womens = rng.integers(0, 2, n)
zip_code = rng.choice(["Urban", "Surburban", "Rural"], n, p=[0.45, 0.40, 0.15])
newbie = rng.integers(0, 2, n)                        # 1 = new customer
channel = rng.choice(["Web", "Phone", "Multichannel"], n, p=[0.4, 0.4, 0.2])

# Randomized experiment: each customer gets one of three random treatments
segment = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
treated = (segment != "No E-Mail").astype(int)

# --- The hidden truth ---
# p0 = chance of visiting WITHOUT an email
p0 = (0.04 + 0.16 * (recency <= 3) + 0.06 * ((recency >= 4) & (recency <= 6))
      + 0.03 * (channel == "Multichannel") + 0.00008 * np.minimum(history, 800))

# tau = how much the email changes that chance (the 'uplift')
tau = 0.01 + 0.10 * newbie + 0.04 * (channel == "Web")
sure_thing = (recency <= 3) & (newbie == 0)
tau = np.where(sure_thing, 0.0, tau)                  # would visit anyway
sleeping_dog = sure_thing & (history > 400)
tau = np.where(sleeping_dog, -0.06, tau)              # email annoys them

visit = rng.binomial(1, np.clip(p0 + treated * tau, 0.01, 0.95))
conversion = visit * rng.binomial(1, 0.07, n)
spend = np.round(conversion * rng.gamma(3, 30, n), 2)

bins = [0, 100, 200, 350, 500, 750, 1000, np.inf]
labels = ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
          "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"]
history_segment = pd.cut(history, bins=bins, labels=labels).astype(str)

df = pd.DataFrame({
    "recency": recency, "history_segment": history_segment, "history": history,
    "mens": mens, "womens": womens, "zip_code": zip_code, "newbie": newbie,
    "channel": channel, "segment": segment, "visit": visit,
    "conversion": conversion, "spend": spend,
})
df.to_csv("sample_hillstrom_like.csv", index=False)

# The truth goes in a SEPARATE file. Real data never has it.
pd.DataFrame({"p0": p0, "true_uplift": tau}).to_csv("sample_truth.csv", index=False)
print("Saved sample_hillstrom_like.csv with", len(df), "rows")