import os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===========================
# Page
# ===========================
st.set_page_config(
    page_title="Maruti Suzuki — Customer Decisive Retention Dashboard",
    page_icon="🚗",
    layout="wide",
)
st.title("🚗 Maruti Suzuki — Customer Decisive Retention Dashboard")
st.caption(
    "Primary: detect the *decision moment* (next 30 days) when a customer is ready to switch/upgrade. "
    "Secondary: churn risk (next 6 months). Works offline; optional LLM Q&A if OPENAI_API_KEY is set."
)

# ===========================
# Columns & configuration
# ===========================
ID_COL = "CustomerID"

TARGET_DECISION = "DecisionReady30D"
TARGET_CHURN = "Churn6M"

# Existing (baseline) features
NUM_COLS_BASE = [
    "Age","IncomeLPA","TenureMonths","CarAgeYears","OdometerKM","ServiceVisits12M",
    "LastServiceDays","Complaints12M","NPS","AppLogins30D","WarrantyMonthsLeft",
    "InsuranceRenewalDueDays","AccessoriesSpend12M","ServiceSpend12M"
]
CAT_COLS_BASE = ["City","Segment","Model","FuelType","AcquisitionChannel","InsuranceProvider"]
BIN_COLS_BASE = ["ConnectedCar","WarrantyActive","DiscountSeeking","CompetitorQuoteSeen","ResaleIntent6M"]

# New (decision) features
NUM_COLS_DECISION = [
    "WebsiteVisits30D","ConfiguratorStarts30D","FinanceCalculatorUses30D","OfferEmailsOpened30D",
    "WhatsAppClicks30D","CallCenterInquiries30D","CompetitorModelSearch30D","DaysSinceLastQuote",
    "QuoteValueLakh"
]
BIN_COLS_DECISION = [
    "BrochureDownloaded30D","TradeInValuation30D","DealerVisit30D","TestDriveRequest30D",
    "PriceDropAlertSubscribed","LoanPreApprovalFlag","FamilyEventFlag","SalaryHikeFlag",
    "QuoteRequestedFlag","DecisiveMomentFlag"
]
CAT_COLS_DECISION = ["PreferredUpgradeModel","IntentStage"]

def load_demo_data() -> pd.DataFrame:
    candidates = [
        "maruti_customer_decisive_demo.csv",
        "data/maruti_customer_decisive_demo.csv",
        "maruti_customer_demo.csv",                 # fallback
        "data/maruti_customer_demo.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    st.error("Demo CSV not found. Upload your CSV in the sidebar, or place maruti_customer_decisive_demo.csv next to the app.")
    st.stop()

def band_from_thresholds(p: float, critical_cut: float, high_cut: float, medium_cut: float) -> str:
    if p >= critical_cut: return "Critical"
    if p >= high_cut: return "High"
    if p >= medium_cut: return "Medium"
    return "Low"

def decisive_playbook(row: pd.Series) -> Tuple[str, list]:
    """
    A simple (transparent) AI-agent playbook.
    Treat 'DecisionReady30D' / 'DecisionScore' as the trigger. Convert intent & context into next-best-actions.
    """
    actions = []
    # Moment-of-truth actions (fast turnaround)
    if int(row.get("DecisiveMomentFlag", 0)) == 1:
        actions.append("⚡ **Instant intercept:** call within 2 hours + WhatsApp brochure + test-drive slot within 24h")

    # Trade-in / buyback
    if int(row.get("TradeInValuation30D", 0)) == 1 or int(row.get("ResaleIntent6M", 0)) == 1:
        actions.append("🔁 **Exchange:** guaranteed buyback quote + doorstep inspection + upgrade path to preferred model")

    # Price / finance sensitivity
    if int(row.get("LoanPreApprovalFlag", 0)) == 1 or int(row.get("FinanceCalculatorUses30D", 0)) >= 2:
        actions.append("💳 **Finance:** pre-approved EMI plan + 0 processing fee + limited-time rate lock")
    if int(row.get("DiscountSeeking", 0)) == 1:
        actions.append("🏷️ **Offer:** personalized accessory pack / service bundle instead of raw discount (protect margin)")

    # Competitor leakage
    if int(row.get("CompetitorQuoteSeen", 0)) == 1 or int(row.get("CompetitorModelSearch30D", 0)) >= 5:
        actions.append("🆚 **Competitive defense:** comparison sheet + matching offer guardrails + transparent TCO (3 years)")

    # Service/NPS rescue before pitch
    if int(row.get("Complaints12M", 0)) >= 2 or int(row.get("NPS", 0)) <= 0:
        actions.insert(0, "🧑‍🔧 **Service recovery first:** concierge escalation + RCA + goodwill coupon, then upgrade pitch")

    # Maintenance triggers
    if int(row.get("LastServiceDays", 0)) >= 220:
        actions.append("📅 **Service hook:** free health check + pick-up & drop; attach upgrade consultation")

    if not actions:
        actions = ["🎁 **Default:** loyalty benefit + upgrade preview + 2 test-drive options this week"]

    headline = "AI Agent: next best action pack (retain at the decision moment)"
    return headline, actions

def local_explain_logreg(pipeline: Pipeline, X_row: pd.DataFrame, X_ref: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
    """
    Local explanation using standardized delta from reference mean for logistic regression:
    contribution ~= coef * (x - mean) in transformed space.
    """
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["pre"]
    Xt = pre.transform(X_row)
    Xref_t = pre.transform(X_ref)
    mean_ref = np.asarray(Xref_t.mean(axis=0)).ravel()
    x = np.asarray(Xt).ravel()
    coef = model.coef_.ravel()
    contrib = coef * (x - mean_ref)

    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(contrib))])

    dfc = pd.DataFrame({"Feature": feat_names, "Contribution": contrib})
    dfc["Abs"] = dfc["Contribution"].abs()
    dfc = dfc.sort_values("Abs", ascending=False).head(topk).drop(columns=["Abs"])
    return dfc

def optional_llm_answer(query: str, context: str) -> Optional[str]:
    api = os.getenv("OPENAI_API_KEY","").strip()
    if not api:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api)
        prompt = (
            "You are an automotive retention analytics assistant. "
            "Answer succinctly using only the provided context (tabular snapshot + notes). "
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

def parse_df_query(q: str, df: pd.DataFrame):
    """
    Lightweight dataframe Q&A (offline):
      - "top 20 decision customers"
      - "average DecisionScore by City"
      - "count where DecisionBand=High and City=Delhi NCR"
    """
    ql = q.strip().lower()
    if not ql:
        return None, "Empty query."

    cols = {c.lower(): c for c in df.columns}

    op = None
    for k in ["count","avg","average","mean","sum","min","max","top","show","list"]:
        if re.search(rf"\b{k}\b", ql):
            op = k
            break
    if op is None:
        op = "show"

    # measure preference: DecisionScore then ChurnScore
    measure = cols.get("decisionscore") or ("DecisionScore" if "DecisionScore" in df.columns else None)
    if measure is None:
        measure = cols.get("churnscore") or ("ChurnScore" if "ChurnScore" in df.columns else None)

    for token in re.findall(r"[a-z_]+", ql):
        if token in cols and pd.api.types.is_numeric_dtype(df[cols[token]]):
            measure = cols[token]
            break

    gby = None
    m = re.search(r"\bby\s+([a-z0-9_ ]+)", ql)
    if m:
        gname = m.group(1).strip()
        gby = cols.get(gname.lower())

    filt = {}
    for pat in [r"where\s+(.+)$", r"filter\s+(.+)$"]:
        mm = re.search(pat, q, flags=re.IGNORECASE)
        if mm:
            tail = mm.group(1)
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

    topn = None
    mm = re.search(r"\btop\s+(\d+)", ql)
    if mm:
        topn = int(mm.group(1))

    # Special: decision customers
    if ("decision" in ql or "ready" in ql) and ("customer" in ql or "customers" in ql) and ("top" in ql or "show" in ql or "list" in ql):
        if "DecisionScore" in work.columns:
            out = work.sort_values("DecisionScore", ascending=False)
            if topn is None: topn = 20
            return out.head(topn), f"Top {topn} customers by DecisionScore"

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

    if topn:
        return work.head(topn), f"Showing first {topn} rows after filters"
    return work.head(50), "Showing first 50 rows after filters"

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.header("Data & Models")

    mode = st.radio("Data source", ["Use demo data (synthetic)", "Upload CSV"], index=0)
    uploaded = None
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    st.divider()
    st.subheader("Model choice")
    model_choice = st.selectbox(
        "Classifier",
        ["Logistic Regression (fast + explainable)", "Random Forest (non-linear)"],
        index=0
    )
    test_size = st.slider("Test size", 0.15, 0.40, 0.25, 0.05)
    seed = st.number_input("Random seed", 1, 9999, 42)

    st.divider()
    st.subheader("Operational thresholds")
    st.caption("These drive triage lists; tune to match call-center capacity and SLA.")

    dec_critical = st.slider("Decision Critical ≥", 0.60, 0.95, 0.80, 0.05)
    dec_high     = st.slider("Decision High ≥",     0.35, 0.85, 0.60, 0.05)
    dec_medium   = st.slider("Decision Medium ≥",   0.15, 0.65, 0.35, 0.05)

    ch_critical  = st.slider("Churn Critical ≥",    0.60, 0.95, 0.75, 0.05)
    ch_high      = st.slider("Churn High ≥",        0.35, 0.85, 0.55, 0.05)
    ch_medium    = st.slider("Churn Medium ≥",      0.15, 0.65, 0.25, 0.05)

    st.divider()
    st.header("How to run")
    st.code("pip install -r requirements.txt\nstreamlit run app_customer_decisive_retention_dashboard.py", language="bash")

# ===========================
# Load data
# ===========================
if mode == "Use demo data (synthetic)":
    df = load_demo_data()
else:
    if not uploaded:
        st.info("Upload a CSV to proceed. Expected columns similar to the demo dataset.")
        st.stop()
    df = pd.read_csv(uploaded)

missing = [c for c in [ID_COL, TARGET_CHURN] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Decision target is optional: if absent, the app runs with a weak label proxy using DecisionIntentScore/DecisiveMomentFlag.
has_decision_target = TARGET_DECISION in df.columns

# ===========================
# Feature set
# ===========================
num_cols = [c for c in (NUM_COLS_BASE + NUM_COLS_DECISION) if c in df.columns]
cat_cols = [c for c in (CAT_COLS_BASE + CAT_COLS_DECISION) if c in df.columns]
bin_cols = [c for c in (BIN_COLS_BASE + BIN_COLS_DECISION) if c in df.columns]

feature_cols = num_cols + cat_cols + bin_cols

work = df[[ID_COL] + feature_cols + ([TARGET_DECISION] if has_decision_target else []) + [TARGET_CHURN]].dropna()

X = work[feature_cols]

# ===========================
# Build pipelines
# ===========================
numeric_features = [c for c in feature_cols if (c in num_cols + bin_cols) and pd.api.types.is_numeric_dtype(work[c])]
categorical_features = [c for c in feature_cols if c in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

def make_model(choice: str, seed: int):
    if choice.startswith("Logistic"):
        return LogisticRegression(max_iter=2500, n_jobs=None)
    return RandomForestClassifier(
        n_estimators=400,
        random_state=int(seed),
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1
    )

# ===========================
# Train decision model
# ===========================
decision_notes = ""
if has_decision_target:
    y_dec = work[TARGET_DECISION].astype(int)
else:
    # Proxy label: "ready" if high intent score AND recent quote/test-drive/dealer-visit
    if "DecisionIntentScore" in df.columns:
        score = work.get("DecisionIntentScore", pd.Series(np.zeros(len(work))))
    else:
        score = pd.Series(np.zeros(len(work)))
    proxy = ((score >= 0.72) & ((work.get("DaysSinceLastQuote", 999) <= 7) | (work.get("TestDriveRequest30D", 0) == 1) | (work.get("DealerVisit30D", 0) == 1))).astype(int)
    y_dec = proxy
    decision_notes = "DecisionReady30D not found → using a proxy label based on intent score + recent activity (demo mode)."

pipe_dec = Pipeline(steps=[("pre", pre), ("model", make_model(model_choice, int(seed)))])
Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(X, y_dec, test_size=float(test_size), random_state=int(seed), stratify=y_dec)
pipe_dec.fit(Xtr_d, ytr_d)
p_dec = pipe_dec.predict_proba(Xte_d)[:,1]
auc_dec = roc_auc_score(yte_d, p_dec) if len(np.unique(yte_d))>1 else float("nan")
ap_dec  = average_precision_score(yte_d, p_dec) if len(np.unique(yte_d))>1 else float("nan")

# ===========================
# Train churn model
# ===========================
y_ch = work[TARGET_CHURN].astype(int)
pipe_ch = Pipeline(steps=[("pre", pre), ("model", make_model(model_choice, int(seed)+17))])
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_ch, test_size=float(test_size), random_state=int(seed)+17, stratify=y_ch)
pipe_ch.fit(Xtr_c, ytr_c)
p_ch = pipe_ch.predict_proba(Xte_c)[:,1]
auc_ch = roc_auc_score(yte_c, p_ch)
ap_ch  = average_precision_score(yte_c, p_ch)

# ===========================
# Score all customers
# ===========================
scored = work.copy()
scored["DecisionScore"] = pipe_dec.predict_proba(X)[:,1]
scored["ChurnScore"] = pipe_ch.predict_proba(X)[:,1]

scored["DecisionBand"] = scored["DecisionScore"].apply(lambda p: band_from_thresholds(p, dec_critical, dec_high, dec_medium))
scored["ChurnBand"]    = scored["ChurnScore"].apply(lambda p: band_from_thresholds(p, ch_critical, ch_high, ch_medium))

# A simple operational trigger (agent "wake-up")
scored["AgentTrigger"] = ((scored["DecisionBand"].isin(["Critical","High"])) | (scored.get("DecisiveMomentFlag", 0) == 1)).astype(int)

# ===========================
# Top KPIs
# ===========================
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers", f"{len(scored):,}")
k2.metric("Decision-ready (label/proxy)", f"{(y_dec.mean()*100):.1f}%")
k3.metric("Decision model ROC-AUC", f"{auc_dec:.3f}" if auc_dec==auc_dec else "NA")
k4.metric("Churn rate", f"{(y_ch.mean()*100):.1f}%")
k5.metric("Churn model ROC-AUC", f"{auc_ch:.3f}")
k6.metric("Agent triggers", f"{(scored['AgentTrigger'].mean()*100):.1f}%")

if decision_notes:
    st.info(decision_notes)

st.divider()

# ===========================
# Tabs
# ===========================
tab1, tab2, tab3 = st.tabs(["⚡ Decision Moment (Primary)", "🧯 Churn Risk (Secondary)", "💬 Q&A (offline + optional LLM)"])

# ---------------------------
# Tab 1: Decision Moment
# ---------------------------
with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("📈 Decision distribution & intent signals")

        fig = plt.figure()
        plt.hist(scored["DecisionScore"], bins=30)
        plt.xlabel("DecisionScore (probability of decision to switch/upgrade in next 30 days)")
        plt.ylabel("Customers")
        st.pyplot(fig, clear_figure=True)

        seg_col = st.selectbox(
            "Decision view by",
            ["City","Model","Segment","FuelType","AcquisitionChannel","PreferredUpgradeModel","IntentStage"],
            index=0,
            key="dec_seg"
        )
        grp = scored.groupby(seg_col)["DecisionScore"].mean().sort_values(ascending=False).head(12)

        fig2 = plt.figure()
        plt.bar(grp.index.astype(str), grp.values)
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Average DecisionScore")
        plt.title(f"Avg decision readiness by {seg_col} (top 12)")
        st.pyplot(fig2, clear_figure=True)

        st.caption("Interpretation: this is a *moment-of-truth* signal. High DecisionScore means the customer is actively evaluating a change now.")

    with right:
        st.subheader("🎯 Ready-to-decide customers (instant interception list)")

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            city_f = st.selectbox("City", ["All"] + sorted(scored["City"].dropna().unique().tolist()), index=0, key="dec_city")
        with f2:
            model_f = st.selectbox("Current Model", ["All"] + sorted(scored["Model"].dropna().unique().tolist()), index=0, key="dec_model")
        with f3:
            band_f = st.selectbox("DecisionBand", ["All","Critical","High","Medium","Low"], index=0, key="dec_band")
        with f4:
            trig_only = st.checkbox("Agent triggers only", value=True)

        view = scored.copy()
        if city_f != "All":
            view = view[view["City"] == city_f]
        if model_f != "All":
            view = view[view["Model"] == model_f]
        if band_f != "All":
            view = view[view["DecisionBand"] == band_f]
        if trig_only:
            view = view[view["AgentTrigger"] == 1]

        view = view.sort_values(["DecisionScore","ChurnScore"], ascending=False)

        cols_show = [
            ID_COL,"City","Model","Segment",
            "CarAgeYears","OdometerKM",
            "WebsiteVisits30D","ConfiguratorStarts30D","DealerVisit30D","TestDriveRequest30D",
            "TradeInValuation30D","LoanPreApprovalFlag","CompetitorQuoteSeen","DaysSinceLastQuote",
            "DecisionScore","DecisionBand","ChurnScore","ChurnBand"
        ]
        cols_show = [c for c in cols_show if c in view.columns]
        st.dataframe(view[cols_show].head(30), use_container_width=True, hide_index=True)

        st.caption("Operational intent: intercept within SLA, before competitor closes.")

    st.divider()
    st.subheader("🔎 Customer drill-down (moment explanation + agent actions)")

    c1, c2 = st.columns([1, 1.25])

    with c1:
        cust = st.selectbox("Select CustomerID", scored[ID_COL].head(2500).tolist(), index=0, key="dec_cust")
        row = scored[scored[ID_COL] == cust].iloc[0]

        st.metric("DecisionScore", f"{row['DecisionScore']:.3f}", help="Probability (0..1). Higher means more ready to decide in next 30 days.")
        st.metric("DecisionBand", row["DecisionBand"])
        st.metric("ChurnScore", f"{row['ChurnScore']:.3f}")
        st.metric("AgentTrigger", "YES" if int(row["AgentTrigger"])==1 else "NO")

        st.write("**Profile snapshot**")
        profile = {
            "City": row.get("City",""),
            "CurrentModel": row.get("Model",""),
            "PreferredUpgradeModel": row.get("PreferredUpgradeModel",""),
            "IntentStage": row.get("IntentStage",""),
            "CarAgeYears": float(row.get("CarAgeYears", np.nan)),
            "OdometerKM": int(row.get("OdometerKM", 0)),
            "LastServiceDays": int(row.get("LastServiceDays", 0)),
            "Complaints12M": int(row.get("Complaints12M", 0)),
            "NPS": int(row.get("NPS", 0)),
            "CompetitorQuoteSeen": int(row.get("CompetitorQuoteSeen", 0)),
            "ResaleIntent6M": int(row.get("ResaleIntent6M", 0)),
        }
        st.write(profile)

        st.write("**Recent intent signals (30D)**")
        signals = {
            "WebsiteVisits30D": int(row.get("WebsiteVisits30D", 0)),
            "ConfiguratorStarts30D": int(row.get("ConfiguratorStarts30D", 0)),
            "BrochureDownloaded30D": int(row.get("BrochureDownloaded30D", 0)),
            "FinanceCalculatorUses30D": int(row.get("FinanceCalculatorUses30D", 0)),
            "DealerVisit30D": int(row.get("DealerVisit30D", 0)),
            "TestDriveRequest30D": int(row.get("TestDriveRequest30D", 0)),
            "TradeInValuation30D": int(row.get("TradeInValuation30D", 0)),
            "LoanPreApprovalFlag": int(row.get("LoanPreApprovalFlag", 0)),
            "DaysSinceLastQuote": int(row.get("DaysSinceLastQuote", 999)),
        }
        st.write(signals)

    with c2:
        st.write("**Local explanation (top contributors)**")
        if model_choice.startswith("Logistic"):
            # find row index in work to pick features
            idx = work.index[work[ID_COL] == cust][0]
            X_row = X.loc[[idx]]
            expl = local_explain_logreg(pipe_dec, X_row=X_row, X_ref=Xtr_d, topk=10)
            st.dataframe(expl, use_container_width=True, hide_index=True)
            st.caption("Contributions are approximate (transformed feature space). Positive contributions increase DecisionScore.")
        else:
            st.info("Local explanations are enabled for Logistic Regression. For Random Forest, use global importances below.")

        headline, actions = decisive_playbook(row)
        st.write("**Agent recommended actions (retain at the instant)**")
        st.write(f"**{headline}**")
        for a in actions:
            st.write(f"- {a}")

        st.divider()
        st.write("**Intervention simulator (simple uplift proxy)**")
        st.caption("For demo: assume the agent action improves retention by reducing DecisionScore and/or ChurnScore. Tune factors per program performance.")
        offer_strength = st.slider("Offer strength", 0.0, 1.0, 0.5, 0.05)
        service_recovery = st.checkbox("Service recovery executed", value=(int(row.get("Complaints12M",0))>=2 or int(row.get("NPS",0))<=0))
        trade_in_guarantee = st.checkbox("Guaranteed buyback / exchange", value=(int(row.get("TradeInValuation30D",0))==1 or int(row.get("ResaleIntent6M",0))==1))

        # uplift model (toy): reduce decision score more when offer+trade-in; reduce churn more when service recovery
        dec_reduction = 0.08*offer_strength + (0.10 if trade_in_guarantee else 0.0) + (0.05 if service_recovery else 0.0)
        ch_reduction  = 0.04*offer_strength + (0.07 if service_recovery else 0.0) + (0.03 if trade_in_guarantee else 0.0)

        new_dec = float(np.clip(row["DecisionScore"]*(1.0 - dec_reduction), 0, 1))
        new_ch  = float(np.clip(row["ChurnScore"]*(1.0 - ch_reduction), 0, 1))

        st.write({
            "DecisionScore_before": float(row["DecisionScore"]),
            "DecisionScore_after": new_dec,
            "ChurnScore_before": float(row["ChurnScore"]),
            "ChurnScore_after": new_ch,
            "Estimated_decision_drop": float(row["DecisionScore"] - new_dec),
        })

    st.divider()
    st.subheader("🧠 Global drivers (what moves decision readiness)")

    if model_choice.startswith("Logistic"):
        try:
            model = pipe_dec.named_steps["model"]
            pre2 = pipe_dec.named_steps["pre"]
            feat_names = pre2.get_feature_names_out()
            coefs = model.coef_.ravel()
            imp = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
            imp["Abs"] = imp["Coefficient"].abs()
            imp = imp.sort_values("Abs", ascending=False).head(20).drop(columns=["Abs"])
            st.dataframe(imp, use_container_width=True, hide_index=True)
            st.caption("For Logistic Regression: positive coefficient increases decision readiness; negative decreases.")
        except Exception as e:
            st.write(f"Could not extract coefficients: {e}")
    else:
        try:
            pre2 = pipe_dec.named_steps["pre"]
            model = pipe_dec.named_steps["model"]
            feat_names = pre2.get_feature_names_out()
            imp = pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
            imp = imp.sort_values("Importance", ascending=False).head(20)
            st.dataframe(imp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.write(f"Could not extract importances: {e}")

# ---------------------------
# Tab 2: Churn risk (secondary)
# ---------------------------
with tab2:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("📊 Churn distribution & segments")

        fig = plt.figure()
        plt.hist(scored["ChurnScore"], bins=30)
        plt.xlabel("ChurnScore (probability of churn in next 6 months)")
        plt.ylabel("Customers")
        st.pyplot(fig, clear_figure=True)

        seg_col = st.selectbox("Churn view by", ["City","Model","Segment","FuelType","AcquisitionChannel","InsuranceProvider"], index=0, key="ch_seg")
        grp = scored.groupby(seg_col)["ChurnScore"].mean().sort_values(ascending=False).head(12)

        fig2 = plt.figure()
        plt.bar(grp.index.astype(str), grp.values)
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Average ChurnScore")
        plt.title(f"Avg churn risk by {seg_col} (top 12)")
        st.pyplot(fig2, clear_figure=True)

    with right:
        st.subheader("🚨 Highest churn-risk customers (triage list)")

        f1, f2, f3 = st.columns(3)
        with f1:
            city_f = st.selectbox("City", ["All"] + sorted(scored["City"].dropna().unique().tolist()), index=0, key="ch_city")
        with f2:
            model_f = st.selectbox("Model", ["All"] + sorted(scored["Model"].dropna().unique().tolist()), index=0, key="ch_model")
        with f3:
            band_f = st.selectbox("ChurnBand", ["All","Critical","High","Medium","Low"], index=0, key="ch_band")

        view = scored.copy()
        if city_f != "All":
            view = view[view["City"] == city_f]
        if model_f != "All":
            view = view[view["Model"] == model_f]
        if band_f != "All":
            view = view[view["ChurnBand"] == band_f]

        view = view.sort_values("ChurnScore", ascending=False)

        cols_show = [
            ID_COL,"City","Model","Segment","LastServiceDays","Complaints12M","NPS",
            "WarrantyActive","CompetitorQuoteSeen","ResaleIntent6M","ChurnScore","ChurnBand",
            "DecisionScore","DecisionBand"
        ]
        cols_show = [c for c in cols_show if c in view.columns]
        st.dataframe(view[cols_show].head(30), use_container_width=True, hide_index=True)

        st.caption("Use churn to plan medium-term retention; use decision moment to intercept *right now*.")

    st.divider()
    st.subheader("🧠 Global drivers (what moves churn)")

    if model_choice.startswith("Logistic"):
        try:
            model = pipe_ch.named_steps["model"]
            pre2 = pipe_ch.named_steps["pre"]
            feat_names = pre2.get_feature_names_out()
            coefs = model.coef_.ravel()
            imp = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
            imp["Abs"] = imp["Coefficient"].abs()
            imp = imp.sort_values("Abs", ascending=False).head(20).drop(columns=["Abs"])
            st.dataframe(imp, use_container_width=True, hide_index=True)
            st.caption("For Logistic Regression: positive coefficient increases churn risk; negative decreases.")
        except Exception as e:
            st.write(f"Could not extract coefficients: {e}")
    else:
        try:
            pre2 = pipe_ch.named_steps["pre"]
            model = pipe_ch.named_steps["model"]
            feat_names = pre2.get_feature_names_out()
            imp = pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
            imp = imp.sort_values("Importance", ascending=False).head(20)
            st.dataframe(imp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.write(f"Could not extract importances: {e}")

# ---------------------------
# Tab 3: Q&A
# ---------------------------
with tab3:
    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Ask questions")
    st.caption(
        "Offline examples: “top 20 decision customers where City=Pune”, "
        "“average DecisionScore by Model”, “count where DecisionBand=Critical”, "
        "“top 10 decision customers where PreferredUpgradeModel=Grand Vitara”"
    )

    def add_msg(role, content):
        st.session_state.history.append({"role": role, "content": content})

    for m in st.session_state.history[-20:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask a question about decision readiness / churn…")

    if q:
        with st.chat_message("user"):
            st.markdown(q)
        add_msg("user", q)

        snap = scored.sort_values("DecisionScore", ascending=False).head(12)[
            [ID_COL,"City","Model","PreferredUpgradeModel","IntentStage",
             "WebsiteVisits30D","ConfiguratorStarts30D","DealerVisit30D","TestDriveRequest30D",
             "TradeInValuation30D","CompetitorQuoteSeen","DaysSinceLastQuote",
             "DecisionScore","DecisionBand","ChurnScore","ChurnBand"]
        ]
        notes = (
            f"Decision model: AUC={auc_dec:.3f} AP={ap_dec:.3f}. "
            f"Churn model: AUC={auc_ch:.3f} AP={ap_ch:.3f}. "
            f"Decision thresholds: Med>={dec_medium}, High>={dec_high}, Crit>={dec_critical}."
        )
        context = notes + "\n\nTop decision snapshot:\n" + snap.to_csv(index=False)

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

        add_msg("assistant", f"Offline: {formula}" + (f"\n\nLLM: {llm_ans}" if llm_ans else ""))
