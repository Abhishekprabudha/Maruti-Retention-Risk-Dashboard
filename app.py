import os, re, io, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
from docx import Document

# ---------- Config ----------
st.set_page_config(page_title="Procurement Control Tower: Contracts + Supplier Ranking", page_icon="🧠", layout="wide")
st.title("🧠 AI/Agentic AI – Autonomous Procurement Control Tower")
st.caption("Upload a contract/RFP (PDF/DOCX) and a supplier spreadsheet (XLSX/CSV). Ask questions, rank suppliers, and demo autonomous decisions.")

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Helpers ----------
def load_pdf(file) -> str:
    txt = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            txt.append(t)
    return "\n".join(txt)

def load_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=800):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def find_best_passages(query, text, topk=3):
    q = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ch in chunk_text(text, 160):
        t = set(re.findall(r"\w+", ch.lower()))
        score = len(q & t)
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:topk]]

def safe_number(x):
    try:
        return float(x)
    except:
        return None

def parse_excel_query(q, df):
    """
    Domain-light parser for quick queries like:
      - sum of UnitPriceUSD where Category=Fibre Cable
      - average LeadTimeDays by Supplier
      - count of Item where Country = Norway
    """
    ql = q.lower()
    op = None
    for k in ["sum","total","avg","average","mean","min","max","count"]:
        if re.search(rf"\b{k}\b", ql):
            op = {"sum":"sum","total":"sum","avg":"mean","average":"mean","mean":"mean",
                  "min":"min","max":"max","count":"count"}[k]
            break
    if op is None:
        op = "sum" if any(c in ql for c in ["revenue","amount","price","value","unitpriceusd"]) else "mean"

    cols = {c.lower(): c for c in df.columns}
    measure = None
    for token in re.findall(r"[A-Za-z_]+", ql):
        if token in cols and pd.api.types.is_numeric_dtype(df[cols[token]]):
            measure = cols[token]; break
    if measure is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        measure = num_cols[0] if num_cols else df.columns[0]

    # Filters
    filt = {}
    patterns = [
        r"where\s+([A-Za-z0-9_ ]+)\s*=\s*([A-Za-z0-9_\-./ ]+)",
        r"([A-Za-z0-9_ ]+)\s*:\s*([A-Za-z0-9_\-./ ]+)"
    ]
    for pat in patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            left, right = m.group(1).strip(), m.group(2).strip()
            key = cols.get(left.lower())
            if key and key in df.columns:
                filt[key] = right

    # Group-by
    gby = None
    gbym = re.search(r"\bby\s+([A-Za-z0-9_ ]+)", ql)
    if gbym:
        gcol_name = gbym.group(1).strip()
        gby = cols.get(gcol_name.lower())

    work = df.copy()
    for k, v in filt.items():
        vn = safe_number(v)
        if vn is not None and pd.api.types.is_numeric_dtype(work[k]):
            work = work[work[k] == vn]
        else:
            work = work[work[k].astype(str).str.lower() == str(v).lower()]

    if gby and gby in work.columns:
        if op == "count":
            out = work.groupby(gby)[measure].count().reset_index(name="count")
        else:
            out = getattr(work.groupby(gby)[measure], op)().reset_index(name=f"{op.upper()}({measure})")
        return out, f"{op.upper()}({measure}) by {gby}"

    if op == "count":
        val = work[measure].count()
        return val, f"COUNT({measure})"

    val = getattr(work[measure], op)()
    return val, f"{op.upper()}({measure})"

def call_llm(query, context_text):
    api = os.getenv("OPENAI_API_KEY", "")
    if not api:
        passages = find_best_passages(query, context_text, topk=2)
        if not passages:
            return "No obvious match found in the document."
        joined = "\n\n".join(passages)
        return f"Closest passages:\n\n{textwrap.shorten(joined, width=1200)}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api)
        prompt = (f"You are a procurement analyst. Use the context to answer the user succinctly. "
                  f"If unknown, say so.\n\nUser: {query}\n\nContext:\n{context_text[:12000]}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM disabled) {e}"

def score_suppliers(df, w_price=0.4, w_lead=0.2, w_ontime=0.1, w_defect=0.1, w_esg=0.1, w_risk=0.1):
    """
    Compute a normalized composite score per supplier. Lower price/lead/defect/risk is better; higher on-time/ESG is better.
    Returns a dataframe sorted by score desc.
    """
    work = df.copy()
    # pick columns if present
    colmap = {
        "UnitPriceUSD":"min",
        "LeadTimeDays":"min",
        "OnTimePct":"max",
        "DefectRatePct":"min",
        "ESGScore":"max",
        "RiskScore":"min"
    }
    # normalize columns 0..1 per column direction
    for c, direction in colmap.items():
        if c in work.columns:
            x = work[c].astype(float)
            if direction == "min":
                work[f"norm_{c}"] = 1 - (x - x.min())/(x.max() - x.min() + 1e-9)
            else:
                work[f"norm_{c}"] = (x - x.min())/(x.max() - x.min() + 1e-9)
        else:
            work[f"norm_{c}"] = 0.5  # neutral if missing

    w = dict(price=w_price, lead=w_lead, ontime=w_ontime, defect=w_defect, esg=w_esg, risk=w_risk)
    work["SupplierScore"] = (
        w["price"]*work["norm_UnitPriceUSD"] +
        w["lead"]*work["norm_LeadTimeDays"] +
        w["ontime"]*work["norm_OnTimePct"] +
        w["defect"]*work["norm_DefectRatePct"] +
        w["esg"]*work["norm_ESGScore"] +
        w["risk"]*work["norm_RiskScore"]
    )

    agg = work.groupby("Supplier", as_index=False).agg({
        "UnitPriceUSD":"mean",
        "LeadTimeDays":"mean",
        "OnTimePct":"mean",
        "DefectRatePct":"mean",
        "ESGScore":"mean",
        "RiskScore":"mean",
        "SupplierScore":"mean",
        "Country":"first"
    }).sort_values("SupplierScore", ascending=False)

    return agg

# ---------- UI: uploads ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Upload Contract / RFP (PDF/DOCX)")
    doc_file = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"], key="docxpdf")

with col2:
    st.subheader("📈 Upload Supplier Data (XLSX/CSV)")
    data_file = st.file_uploader("Upload XLSX/CSV", type=["xlsx","csv"], key="csvxlsx")

doc_text = ""
if doc_file:
    if doc_file.name.lower().endswith(".pdf"):
        doc_text = load_pdf(doc_file)
    else:
        doc_text = load_docx(doc_file)

df = None
if data_file:
    if data_file.name.lower().endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)

# ---------- Supplier Ranking Panel ----------
st.divider()
st.subheader("🏆 Supplier Ranking (click-to-rank)")

if df is not None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: w_price = st.slider("Weight: Price", 0.0, 1.0, 0.40, 0.05)
    with c2: w_lead  = st.slider("Lead Time", 0.0, 1.0, 0.20, 0.05)
    with c3: w_ont   = st.slider("On-Time %", 0.0, 1.0, 0.10, 0.05)
    with c4: w_def   = st.slider("Defect %", 0.0, 1.0, 0.10, 0.05)
    with c5: w_esg   = st.slider("ESG Score", 0.0, 1.0, 0.10, 0.05)
    with c6: w_risk  = st.slider("Risk (lower better)", 0.0, 1.0, 0.10, 0.05)

    total = w_price + w_lead + w_ont + w_def + w_esg + w_risk
    if abs(total - 1.0) > 1e-6:
        st.warning(f"Weights sum to {total:.2f}. Consider normalizing to 1 for cleaner interpretation.")

    rank_df = score_suppliers(
        df,
        w_price=w_price, w_lead=w_lead, w_ontime=w_ont,
        w_defect=w_def, w_esg=w_esg, w_risk=w_risk
    )
    st.dataframe(rank_df.head(20), use_container_width=True)
else:
    st.info("Upload a supplier dataset to enable ranking. Try the demo Excel in the sidebar →")

# ---------- Chat Input ----------
st.divider()
query = st.chat_input("Ask a question about the contract/RFP or supplier spreadsheet…")

def add_msg(role, content):
    st.session_state.history.append({"role": role, "content": content})

chat = st.container()
with chat:
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        add_msg("user", query)

        answer_parts = []

        if doc_text:
            ans = call_llm(query, doc_text)
            answer_parts.append(f"**Contract/RFP insight**\n\n{ans}")

        if df is not None:
            try:
                val, formula = parse_excel_query(query, df)
                if isinstance(val, pd.DataFrame):
                    st.dataframe(val)
                    answer_parts.append(f"**Supplier calc** — inferred: `{formula}`")
                else:
                    answer_parts.append(f"**Supplier calc** — inferred: `{formula}` → **{val:,.2f}**")
            except Exception as e:
                answer_parts.append(f"Supplier calc error: {e}")

        if not answer_parts:
            answer_parts.append("Please upload a contract/RFP or a supplier spreadsheet first.")

        with st.chat_message("assistant"):
            st.markdown("\n\n".join(answer_parts))
        add_msg("assistant", "\n\n".join(answer_parts))

# ---------- Sidebar help ----------
with st.sidebar:
    st.header("Demo Files & Tips")
    st.markdown("""
**Download demo files**
- [Supplier Excel demo](sandbox:/mnt/data/procurement_demo.xlsx)
- [RFP/Contract PDF demo](sandbox:/mnt/data/RFP_example.pdf)

**Prompts to try**
- “Extract delivery and warranty terms from the contract.”
- “List ESG requirements and minimum scores.”
- “Average UnitPriceUSD by Supplier for Fibre Cable.”
- “Sum of UnitPriceUSD where Country = Norway and Category = ‘RAN Antenna’.”
- “LeadTimeDays by Supplier; sort ascending.”

**Notes**
- LLM is optional. Set `OPENAI_API_KEY` for semantic answers.
- Spreadsheet Q&A and Supplier Ranking work fully offline.
""")
