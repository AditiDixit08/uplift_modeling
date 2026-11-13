"""app.py - the screen the user sees. Run with: streamlit run app.py"""
import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="Who Should Get the Discount?", layout="wide")
st.title("Who Should Get the Discount?")
st.caption("Uplift modeling: find the people the offer will actually change.")

# ---------- Sidebar: choose data ----------
source = st.sidebar.radio("Data", ["Practice file", "Upload my CSV"])
treat_col = st.sidebar.text_input("Treatment column", "segment")
out_col = st.sidebar.text_input("Outcome column (0/1)", "visit")
top = st.sidebar.slider("Contact the top X% of customers", 5, 100, 20, step=5) / 100

if source == "Upload my CSV":
    file = st.sidebar.file_uploader("CSV file", type="csv")
    if file is None:
        st.info("Upload a CSV to start.")
        st.stop()
    key, path = file.name, io.BytesIO(file.getvalue())
else:
    key, path = "practice", "sample_hillstrom_like.csv"


@st.cache_data(show_spinner="Training models... (a few seconds)")
def cached_run(key, _path, treat_col, out_col):
    return run_pipeline(_path, treat_col, out_col)


try:
    out = cached_run(key, path, treat_col, out_col)
except Exception as e:
    st.error(f"Could not run: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. The experiment", "2. Models", "3. Segments", "4. Who to contact"])

# ---------- Tab 1: the experiment ----------
with tab1:
    e = out["effect"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Bought, got the offer", f"{e['treated_rate']:.1%}")
    c2.metric("Bought, no offer", f"{e['control_rate']:.1%}")
    c3.metric("Overall effect of the offer", f"{e['ate']:+.1%}")
    st.write(f"95% range: {e['ci_low']:+.1%} to {e['ci_high']:+.1%}. "
             f"People: {e['n_treated']} treated, {e['n_control']} control.")
    bal = out["balance"]
    if bal["ok"].all():
        st.success("Randomization check passed: treated and control groups look alike.")
    else:
        st.warning("Some columns differ between the groups. The test may not be random.")
        st.dataframe(bal[~bal["ok"]], hide_index=True)

# ---------- Tab 2: models ----------
with tab2:
    st.write("Real uplift among the top 20% of people each model picks "
             "(measured on customers the model never saw).")
    st.dataframe(pd.DataFrame(out["results"]), hide_index=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (x, q) in out["curves"].items():
        ax.plot(x * 100, q, label=name, linestyle="--" if name == "Random" else "-")
    ax.set_xlabel("% of customers contacted (best first)")
    ax.set_ylabel("Extra buyers gained")
    ax.set_title("Qini curve")
    ax.legend()
    st.pyplot(fig)

# ---------- Tab 3: segments ----------
with tab3:
    seg_table = out["segment_table"]
    st.dataframe(seg_table.style.format({"avg_buy_chance": "{:.1%}", "avg_uplift": "{:+.1%}",
                                         "share": "{:.1%}"}))
    st.bar_chart(seg_table["people"])
    st.write("Persuadables: the offer works. Sure Things: they buy anyway. "
             "Lost Causes: they do not buy either way. Sleeping Dogs: the offer hurts.")

# ---------- Tab 4: who to contact ----------
with tab4:
    targets = out["targets"].sort_values("uplift", ascending=False)
    n_pick = int(len(targets) * top)
    picked = targets.head(n_pick)
    st.metric(f"Customers in the top {top:.0%}", n_pick)
    st.write(f"Average predicted uplift of this group: {picked['uplift'].mean():+.1%}")
    st.dataframe(picked.head(100))
    st.download_button("Download target list (CSV)", picked.to_csv().encode("utf-8"),
                       "target_list.csv", "text/csv")