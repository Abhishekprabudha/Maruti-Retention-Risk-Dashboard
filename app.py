import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page ----------------
st.set_page_config(page_title="Maruti Suzuki — Retention Risk Dashboard", page_icon="🚗", layout="wide")
st.title("🚗 Maruti Suzuki — Retention Risk Dashboard")
st.caption("Detect customers at risk of churn (next 6 months), explain drivers, and simulate retention actions. Works offline; optional LLM Q&A if OPENAI_API_KEY is set.")

# ---------------- Helpers ----------------
TARGET_COL = "Churn6M"
ID_COL = "CustomerID"

NUM_COLS_DEFAULT = [
    "Age","IncomeLPA","TenureMonths","CarAgeYears","OdometerKM","ServiceVisits12M",
    "LastServiceDays","Complaints12M","NPS","AppLogins30D","WarrantyMonthsLeft",
    "InsuranceRenewalDueDays","AccessoriesSpend12M","ServiceSpend12M"
]
CAT_COLS_DEFAULT = ["City","Segment","Model","FuelType","AcquisitionChannel","InsuranceProvider"]
BIN_COLS_DEFAULT = ["ConnectedCar","WarrantyActive","DiscountSeeking","CompetitorQuoteSeen","ResaleIntent6M"]

def load_demo_data() -> pd.DataFrame:
    # The demo CSV is expected next to this app or in the working directory.
    candidates = ["maruti_customer_demo.csv", "data/maruti_customer_demo.csv"]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    st.error("Demo CSV not found. Upload your CSV in the sidebar, or place maruti_customer_demo.csv next to app.")
    st.stop()

def add_risk_band(p: pd.Series) -> pd.Series:
    # Tuned for operational triage; adjust via UI threshold sliders if desired
    bands = pd.cut(
        p,
        bins=[-1, 0.25, 0.55, 0.75, 1.01],
        labels=["Low","Medium","High","Critical"]
    )
    return bands.astype(str)

def retention_playbook(row: pd.Series) -> Tuple[str, list]:
    """Simple rule-based playbook to convert drivers into actions."""
    actions = []
    headline = "Proactive retention outreach"
    if row.get("LastServiceDays", 0) >= 220:
        actions.append("📅 Service reminder + priority slot (pick-up & drop)")
    if row.get("WarrantyActive", 1) == 0:
        actions.append("🛡️ Extended warranty offer (limited-time)")
    if row.get("Complaints12M", 0) >= 2 or row.get("NPS", 0) <= 0:
        actions.append("🧑‍🔧 Concierge escalation + RCA + goodwill coupon")
    if row.get("CompetitorQuoteSeen", 0) == 1:
        actions.append("🏷️ Match competitor quote + transparency breakdown")
    if row.get("InsuranceRenewalDueDays", 999) <= 30:
        actions.append("🧾 Insurance renewal bundle (service + add-ons)")
    if row.get("ResaleIntent6M", 0) == 1:
        actions.append("🔁 Buyback / exchange program with upgrade path")
    if row.get("AppLogins30D", 0) == 0:
        actions.append("📲 App reactivation: benefits + connected features demo")
    if not actions:
        actions = ["🎁 Loyalty benefit: free checkup + accessory voucher"]
    return headline, actions

def parse_df_query(q: str, df: pd.DataFrame):
    """
    Lightweight dataframe Q&A (offline):
      - "top 20 risk customers"
      - "average churn score by City"
      - "count where RiskBand=High and City=Delhi NCR"
      - "show customers where Model=Brezza and RiskBand in (High,Critical)"
    """
    ql = q.strip().lower()
    if not ql:
        return None, "Empty query."

    cols = {c.lower(): c for c in df.columns}

    # Operation detection
    op = None
    for k in ["count","avg","average","mean","sum","min","max","top","show","list"]:
        if re.search(rf"\b{k}\b", ql):
            op = k
            break
    if op is None:
        op = "show"

    # Measure detection (defaults to ChurnScore if present)
    measure = cols.get("churnscore", None) or ("ChurnScore" if "ChurnScore" in df.columns else None)
    for token in re.findall(r"[a-z_]+", ql):
        if token in cols and pd.api.types.is_numeric_dtype(df[cols[token]]):
            measure = cols[token]
            break

    # Group-by detection
    gby = None
    m = re.search(r"\bby\s+([a-z0-9_ ]+)", ql)
    if m:
        gname = m.group(1).strip()
        gby = cols.get(gname.lower())

    # Filters: "where A=B" and "A=B"
    filt = {}
    for pat in [r"where\s+(.+)$", r"filter\s+(.+)$"]:
        mm = re.search(pat, q, flags=re.IGNORECASE)
        if mm:
            tail = mm.group(1)
            # split on and/or (simple)
            parts = re.split(r"\band\b", tail, flags=re.IGNORECASE)
            for p in parts:
                if "=" in p:
                    left, right = p.split("=", 1)
                    left = left.strip()
                    right = right.strip().strip("'\"")
                    key = cols.get(left.lower())
                    if key:
                        filt[key] = right
            break
    # also parse standalone A=B tokens
    for mm in re.finditer(r"([A-Za-z0-9_ ]+)\s*=\s*([A-Za-z0-9_\-./ ]+)", q):
        left, right = mm.group(1).strip(), mm.group(2).strip().strip("'\"")
        key = cols.get(left.lower())
        if key:
            filt[key] = right

    work = df.copy()
    for k, v in filt.items():
        if k not in work.columns:
            continue
        if pd.api.types.is_numeric_dtype(work[k]):
            try:
                work = work[work[k] == float(v)]
            except:
                pass
        else:
            work = work[work[k].astype(str).str.lower() == str(v).lower()]

    # Top N
    topn = None
    mm = re.search(r"\btop\s+(\d+)", ql)
    if mm:
        topn = int(mm.group(1))

    # Special: risk customers
    if "risk" in ql and ("customer" in ql or "customers" in ql) and ("top" in ql or "show" in ql or "list" in ql):
        if "ChurnScore" in work.columns:
            out = work.sort_values("ChurnScore", ascending=False)
            if topn is None: topn = 20
            return out.head(topn), f"Top {topn} customers by ChurnScore"

    # Aggregations
    if op in ["count"]:
        if gby and gby in work.columns:
            out = work.groupby(gby).size().reset_index(name="Count")
            return out.sort_values("Count", ascending=False), f"COUNT by {gby}"
        return int(len(work)), "COUNT(rows)"

    if op in ["avg","average","mean","sum","min","max"]:
        if measure is None:
            return None, "No numeric measure found to aggregate."
        func = {"avg":"mean","average":"mean","mean":"mean","sum":"sum","min":"min","max":"max"}[op]
        if gby and gby in work.columns:
            out = getattr(work.groupby(gby)[measure], func)().reset_index(name=f"{func.upper()}({measure})")
            return out.sort_values(out.columns[-1], ascending=False), f"{func.upper()}({measure}) by {gby}"
        val = getattr(work[measure], func)()
        return float(val), f"{func.upper()}({measure})"

    # Default show/list
    if topn:
        return work.head(topn), f"Showing first {topn} rows after filters"
    return work.head(50), "Showing first 50 rows after filters"

def optional_llm_answer(query: str, context: str) -> Optional[str]:
    api = os.getenv("OPENAI_API_KEY","").strip()
    if not api:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api)
        prompt = (
            "You are a retention analytics assistant for an automotive OEM. "
            "Answer succinctly using the context (tabular snapshot + metric notes). "
            "If not supported, say so.\n\n"
            f"User: {query}\n\nContext:\n{context[:12000]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error) {e}"

def local_explain_logreg(pipeline: Pipeline, X_row: pd.DataFrame, X_ref: pd.DataFrame, topk: int = 6) -> pd.DataFrame:
    """
    Local explanation using standardized delta from reference mean for logistic regression:
    contribution ~= coef * (x - mean) in transformed space.
    """
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["pre"]
    # transformed matrices
    Xt = pre.transform(X_row)
    Xref_t = pre.transform(X_ref)
    mean_ref = np.asarray(Xref_t.mean(axis=0)).ravel()
    x = np.asarray(Xt).ravel()
    coef = model.coef_.ravel()
    contrib = coef * (x - mean_ref)
    # feature names
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(contrib))])
    dfc = pd.DataFrame({"Feature":feat_names, "Contribution":contrib})
    dfc["Abs"] = dfc["Contribution"].abs()
    dfc = dfc.sort_values("Abs", ascending=False).head(topk).drop(columns=["Abs"])
    return dfc

# ---------------- Sidebar: data ----------------
with st.sidebar:
    st.header("Data & Model")
    mode = st.radio("Data source", ["Use demo data (synthetic)", "Upload CSV"], index=0)

    uploaded = None
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    model_choice = st.selectbox("Model", ["Logistic Regression (fast + explainable)", "Random Forest (non-linear)"], index=0)
    test_size = st.slider("Test size", 0.15, 0.40, 0.25, 0.05)
    seed = st.number_input("Random seed", 1, 9999, 42)

    st.divider()
    st.subheader("Operational thresholds")
    critical_cut = st.slider("Critical risk ≥", 0.60, 0.95, 0.75, 0.05)
    high_cut = st.slider("High risk ≥", 0.35, 0.80, 0.55, 0.05)
    medium_cut = st.slider("Medium risk ≥", 0.15, 0.60, 0.25, 0.05)

    st.caption("RiskBand is derived from these thresholds (Critical/High/Medium/Low).")

# ---------------- Load data ----------------
if mode == "Use demo data (synthetic)":
    df = load_demo_data()
else:
    if not uploaded:
        st.info("Upload a CSV to proceed. Expected columns similar to the demo dataset.")
        st.stop()
    df = pd.read_csv(uploaded)

if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' missing. For real deployment, set it using your churn definition (e.g., no service > 12 months).")
    st.stop()
if ID_COL not in df.columns:
    st.error(f"ID column '{ID_COL}' missing.")
    st.stop()

# ---------------- Feature set ----------------
num_cols = [c for c in NUM_COLS_DEFAULT if c in df.columns]
cat_cols = [c for c in CAT_COLS_DEFAULT if c in df.columns]
bin_cols = [c for c in BIN_COLS_DEFAULT if c in df.columns]

feature_cols = num_cols + cat_cols + bin_cols
work = df[[ID_COL] + feature_cols + [TARGET_COL]].dropna()

X = work[feature_cols]
y = work[TARGET_COL].astype(int)

# ---------------- Build model pipeline ----------------
numeric_features = [c for c in feature_cols if c in num_cols + bin_cols and pd.api.types.is_numeric_dtype(work[c])]
categorical_features = [c for c in feature_cols if c in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

if model_choice.startswith("Logistic"):
    model = LogisticRegression(max_iter=2000, n_jobs=None)
else:
    model = RandomForestClassifier(
        n_estimators=350,
        random_state=int(seed),
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1
    )

pipe = Pipeline(steps=[("pre", pre), ("model", model)])

# ---------------- Train/eval ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(seed), stratify=y)

pipe.fit(X_train, y_train)
p_test = pipe.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, p_test)
ap = average_precision_score(y_test, p_test)

# Confusion at 0.5 (for reference)
yhat = (p_test >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()

# Score all customers
p_all = pipe.predict_proba(X)[:,1]
scored = work.copy()
scored["ChurnScore"] = p_all

# Apply user thresholds
def band(p):
    if p >= critical_cut: return "Critical"
    if p >= high_cut: return "High"
    if p >= medium_cut: return "Medium"
    return "Low"

scored["RiskBand"] = scored["ChurnScore"].apply(band)

# ---------------- Top KPIs ----------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(scored):,}")
k2.metric("Observed churn rate", f"{y.mean()*100:.1f}%")
k3.metric("Model ROC-AUC", f"{auc:.3f}")
k4.metric("Avg Precision", f"{ap:.3f}")
k5.metric("Critical+High", f"{(scored['RiskBand'].isin(['Critical','High']).mean()*100):.1f}%")

st.divider()

# ---------------- Dashboard layout ----------------
left, right = st.columns([1.25, 1])

with left:
    st.subheader("📊 Risk distribution & segments")

    # Histogram of churn scores
    fig = plt.figure()
    plt.hist(scored["ChurnScore"], bins=30)
    plt.xlabel("ChurnScore (probability of churn in next 6 months)")
    plt.ylabel("Customers")
    st.pyplot(fig, clear_figure=True)

    # Risk by key segment
    seg_col = st.selectbox("Segment view", ["City","Model","Segment","FuelType","AcquisitionChannel","InsuranceProvider"], index=0)
    grp = scored.groupby(seg_col)["ChurnScore"].mean().sort_values(ascending=False).head(12)
    fig2 = plt.figure()
    plt.bar(grp.index.astype(str), grp.values)
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Average ChurnScore")
    plt.title(f"Avg risk by {seg_col} (top 12)")
    st.pyplot(fig2, clear_figure=True)

with right:
    st.subheader("🚨 Highest-risk customers (triage list)")

    f1, f2, f3 = st.columns(3)
    with f1:
        city_f = st.selectbox("City", ["All"] + sorted(scored["City"].dropna().unique().tolist()), index=0)
    with f2:
        model_f = st.selectbox("Model", ["All"] + sorted(scored["Model"].dropna().unique().tolist()), index=0)
    with f3:
        band_f = st.selectbox("RiskBand", ["All","Critical","High","Medium","Low"], index=0)

    view = scored.copy()
    if city_f != "All":
        view = view[view["City"] == city_f]
    if model_f != "All":
        view = view[view["Model"] == model_f]
    if band_f != "All":
        view = view[view["RiskBand"] == band_f]

    view = view.sort_values("ChurnScore", ascending=False)
    st.dataframe(view[[ID_COL,"City","Model","Segment","LastServiceDays","Complaints12M","NPS","WarrantyActive","CompetitorQuoteSeen","ResaleIntent6M","ChurnScore","RiskBand"]].head(30),
                 use_container_width=True, hide_index=True)

    st.caption("Tip: start with Critical → resolve issues, then High → offer service bundles and loyalty incentives.")

st.divider()

# ---------------- Customer drill-down ----------------
st.subheader("🔎 Customer drill-down (explain + actions)")

c1, c2 = st.columns([1, 1.2])
with c1:
    cust = st.selectbox("Select CustomerID", scored[ID_COL].head(2000).tolist(), index=0)  # avoid huge dropdown
    row = scored[scored[ID_COL] == cust].iloc[0]

    st.metric("ChurnScore", f"{row['ChurnScore']:.3f}", help="Probability (0..1). Higher means higher churn risk.")
    st.metric("RiskBand", row["RiskBand"])
    st.write("**Key profile**")
    st.write({
        "City": row.get("City",""),
        "Model": row.get("Model",""),
        "Segment": row.get("Segment",""),
        "CarAgeYears": float(row.get("CarAgeYears", np.nan)),
        "OdometerKM": int(row.get("OdometerKM", 0)),
        "LastServiceDays": int(row.get("LastServiceDays", 0)),
        "ServiceVisits12M": int(row.get("ServiceVisits12M", 0)),
        "Complaints12M": int(row.get("Complaints12M", 0)),
        "NPS": int(row.get("NPS", 0)),
        "WarrantyActive": int(row.get("WarrantyActive", 0)),
        "CompetitorQuoteSeen": int(row.get("CompetitorQuoteSeen", 0)),
        "ResaleIntent6M": int(row.get("ResaleIntent6M", 0)),
    })

with c2:
    st.write("**Local driver explanation**")
    if model_choice.startswith("Logistic"):
        # local explanation against training reference
        X_row = X.loc[[work.index[work[ID_COL] == cust][0]]]
        expl = local_explain_logreg(pipe, X_row=X_row, X_ref=X_train, topk=8)
        st.dataframe(expl, use_container_width=True, hide_index=True)
        st.caption("Contributions are approximate in transformed feature space; positive contributions increase churn risk.")
    else:
        st.info("Local per-customer explanations are enabled for Logistic Regression. For Random Forest, use the global importances below.")

    headline, actions = retention_playbook(row)
    st.write("**Recommended retention actions**")
    st.write(f"**{headline}**")
    for a in actions:
        st.write(f"- {a}")

st.divider()

# ---------------- Global drivers ----------------
st.subheader("🧠 Global risk drivers (what moves churn)")

if model_choice.startswith("Logistic"):
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    try:
        feat_names = pre.get_feature_names_out()
        coefs = model.coef_.ravel()
        imp = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
        imp["Abs"] = imp["Coefficient"].abs()
        imp = imp.sort_values("Abs", ascending=False).head(18).drop(columns=["Abs"])
        st.dataframe(imp, use_container_width=True, hide_index=True)
        st.caption("For Logistic Regression, positive coefficient increases churn risk; negative decreases.")
    except Exception as e:
        st.write(f"Could not extract coefficients: {e}")
else:
    # crude feature importance using impurity + feature names (best-effort)
    try:
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
        feat_names = pre.get_feature_names_out()
        imp = pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(20)
        st.dataframe(imp, use_container_width=True, hide_index=True)
    except Exception as e:
        st.write(f"Could not extract importances: {e}")

st.divider()

# ---------------- Query / Chat ----------------
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("💬 Ask questions (data Q&A + optional LLM)")
st.caption("Offline queries work without API key. Examples: “top 20 risk customers where City=Delhi NCR”, “average ChurnScore by Model”, “count where RiskBand=Critical”")

def add_msg(role, content):
    st.session_state.history.append({"role": role, "content": content})

for m in st.session_state.history[-20:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask a question about retention risk data…")

if q:
    with st.chat_message("user"):
        st.markdown(q)
    add_msg("user", q)

    # Make a compact context snapshot for optional LLM
    snap = scored.sort_values("ChurnScore", ascending=False).head(12)[
        [ID_COL,"City","Model","LastServiceDays","Complaints12M","NPS","WarrantyActive","CompetitorQuoteSeen","ResaleIntent6M","ChurnScore","RiskBand"]
    ]
    notes = f"Metrics: AUC={auc:.3f}, AP={ap:.3f}. Thresholds: Medium>={medium_cut}, High>={high_cut}, Critical>={critical_cut}."
    context = notes + "\n\nTop-risk snapshot:\n" + snap.to_csv(index=False)

    df_ans, formula = parse_df_query(q, scored)
    llm_ans = optional_llm_answer(q, context)

    with st.chat_message("assistant"):
        if isinstance(df_ans, pd.DataFrame):
            st.dataframe(df_ans, use_container_width=True, hide_index=True)
            st.markdown(f"**Calc (offline):** `{formula}`")
        elif df_ans is not None:
            st.markdown(f"**Calc (offline):** `{formula}` → **{df_ans:,.4f}**")
        else:
            st.markdown(f"**Calc (offline):** {formula}")

        if llm_ans:
            st.markdown("---")
            st.markdown(f"**LLM insight:**\n\n{llm_ans}")

    # store a short assistant message (avoid storing full dataframes)
    add_msg("assistant", f"Offline: {formula}" + (f"\n\nLLM: {llm_ans}" if llm_ans else ""))

with st.sidebar:
    st.divider()
    st.header("How to run")
    st.code("pip install -r requirements.txt\nstreamlit run app_retention_risk_dashboard.py", language="bash")
