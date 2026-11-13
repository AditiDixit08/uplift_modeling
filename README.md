Who Should Get the Discount? (Uplift Modeling)


Predicts who will buy because of an offer, not just who will buy. Upload a randomized email test and the app ranks customers by how much the offer changes their behaviour, then sorts them into Persuadables, Sure Things, Lost Causes and Sleeping Dogs.

Live demo: https://8dmkslwofxasearzetsyw5.streamlit.app/



The problem


Sending a discount to everyone costs money. Some people would have bought anyway, and a few get annoyed and leave. A normal "will they buy?" model ranks exactly these people first. The real question is who will buy because of the email.

How it works


1. Load a randomized experiment (a coin flip decided who got the email).
2. Measure the overall effect (treated rate minus control rate) and check the groups look alike.
3. Train a baseline model, a T-Learner and an X-Learner.
4. Score every held-out customer and evaluate with a Qini curve and uplift in the top 20%.
5. Split customers into the four types and download a target list.

The models

| Model | What it does |
|-------|--------------|
| Baseline | Predicts who will buy and ignores the offer. Kept on purpose to show why it misleads |
| T-Learner | One model for treated people, one for control, then subtract their predictions |
| X-Learner | A T-Learner plus a second round that learns from each model's gaps and blends the results |

Both uplift models are written by hand with scikit-learn (HistGradientBoosting), about 60 lines in total, so every line can be explained.

The four customer types


| Type | Buys without offer? | Buys with offer? | Action |
|------|---------------------|------------------|--------|
| Persuadable | No | Yes | Send the offer |
| Sure Thing | Yes | Yes | Skip, they buy anyway |
| Lost Cause | No | No | Skip |
| Sleeping Dog | Yes | No | Never contact |

Checks built in

- Data leakage protection: result columns (visit, conversion, spend) are removed from the inputs automatically.
- Randomization check: the standardized mean difference is computed for every input column, and the app warns if the groups differ.
- Stratified train/test split, so rare buyers stay balanced in both parts.
- Evaluation only on customers the models never saw.

Results


Practice dataset with planted truth (64,000 customers, built to match the Hillstrom format):

| Model | Uplift in top 20% | Qini score |
|-------|-------------------|------------|
| Random | 6.4% | -0.0007 |
| Baseline | 8.8% | 0.0053 |
| T-Learner | 14.0% | 0.0095 |
| X-Learner | 13.4% | 0.0097 |

The overall effect of the email on this file is +7.4 points (19.3% vs 11.9%). On the practice file, T-Learner and X-Learner scores match the planted true uplift at 0.94 and 0.96, against 0.38 for the baseline. Real data has no truth file, so this check only works here.

Real datasets:

| Dataset | Rows | Overall effect | Uplift top 20% (best model) | Random top 20% |
|---------|------|----------------|-----------------------------|----------------|
| Hillstrom, visit | | | | |
| Hillstrom, conversion | | | | |
| Criteo sample (optional) | | | | |

Tech stack
Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, Streamlit

Project structure

    make_sample_data.py   builds a practice dataset with planted customer types
    prep.py               loading, leakage removal, overall effect, balance check, split
    models.py             baseline, T-Learner and X-Learner
    evaluate.py           Qini curve, Qini score, uplift at top 20%
    segments.py           the four customer types
    pipeline.py           runs everything in one function
    app.py                Streamlit web app

Run locally


    pip install -r requirements.txt
    python make_sample_data.py
    streamlit run app.py

No API key is needed. Everything runs on your own machine.

To use your own file, choose Upload my CSV in the sidebar and type the treatment column and the 0/1 outcome column. A text treatment column works if the control value is "No E-Mail", and a 0/1 column works too.

Limitations

- Needs data from a randomized experiment. Without randomization the groups differ for other reasons and uplift numbers cannot be trusted.
- Small segments are noisy, and rare outcomes like purchases make results unstable.
- The segment cut-offs (0.02 and the median buy chance) are design choices and can be tuned.
- Measures visits or purchases, not profit.
- Real data may have very few Sleeping Dogs, or none. That is a finding, not a bug.

Author
Aditi Dixit
