# streamlit_app.py
"""
RIGVEDIT — JD <> Resume Matcher (Streamlit)
Drop-in replacement implementing:
 - dark theme & readable labels (CSS injection)
 - logo display (assets/images/logo.png)
 - PDF/DOCX text extraction (pdfplumber, python-docx)
 - Skill extraction (yake + spaCy noun chunks)
 - Semantic scoring with SBERT (sentence-transformers) and TF-IDF fallback
 - Experience & education heuristics
 - Match scoring and CSV download
 - Top-3 Plotly bar chart (prettified)
"""

import os
import re
import io
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Text extraction
try:
    import pdfplumber
except Exception:
    pdfplumber = None
from docx import Document

# NLP / SKILL extraction
import yake
import spacy

# Semantic embeddings (SBERT) with fallback to TF-IDF
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Plotly for the top-3 chart
import plotly.graph_objects as go

# Optional: rapidfuzz (not required)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# -------------------------
# Config & CSS
# -------------------------
st.set_page_config(page_title="RIGVEDIT — JD Resume Matcher", layout="wide")

# Inject CSS to ensure labels/browse text are visible on dark background.
st.markdown(
    """
    <style>
    /* Body */
    .stApp { background: linear-gradient(180deg,#061022 0%, #071427 100%); color: #e6eef6; }
    /* Headline & header */
    .app-header { display:flex; gap:18px; align-items:center; margin-bottom:8px; }
    .app-title { font-size:34px; font-weight:800; color:#f2fbff; margin:0; }
    .app-sub { color:#9fbad0; margin:0; font-size:14px }
    /* Make file uploader readable */
    [data-testid="stFileUploader"] { color: #d0e7f8 !important; }
    /* Buttons */
    .stButton>button, [data-testid="stFileUploadButton"] button { background: linear-gradient(90deg,#14d3a5,#1ea7ff) !important; color:#08111a !important; font-weight:700; border-radius:12px !important; padding:8px 14px !important; }
    /* Small hints + uploader descriptions */
    .stText, .css-1lsmgbg, .css-1v3fvcr, .css-1d391kg { color: #cfe7ff !important; opacity:0.95; }
    /* Table */
    table.dataframe thead th, table.dataframe tbody td { color: #dbe9f5 !important; background: rgba(255,255,255,0.01) !important; }
    /* Download button style for anchor fallback */
    .download-csv-btn { background: linear-gradient(90deg,#14d3a5,#1ea7ff); color:#08111a; padding:10px 16px; border-radius:12px; font-weight:700; text-decoration:none; display:inline-block; }
    /* Card container look */
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.01)); border-radius:14px; padding:18px; box-shadow: 0 10px 30px rgba(2,6,23,0.6); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities: text extraction
# -------------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber if available."""
    if not pdfplumber:
        return ""
    try:
        text = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
        return " ".join(text)
    except Exception:
        return ""


def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(b)
        tmp.flush()
        tmp.close()
        doc = Document(tmp.name)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        os.unlink(tmp.name)
        return " ".join(paragraphs)
    except Exception:
        return ""


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# -------------------------
# Skill extraction (YAKE + noun chunks)
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner"])

def extract_skills_from_jd(jd_text: str, topk: int = 40):
    if not jd_text:
        return []
    kw_extractor = yake.KeywordExtractor(top=topk, stopwords=None)
    kw = [k for k, score in kw_extractor.extract_keywords(jd_text)]
    doc = nlp(jd_text)
    noun_chunks = [c.text.lower().strip() for c in doc.noun_chunks if len(c.text) > 2]
    skills = []
    for s in kw + noun_chunks:
        s = s.lower().strip()
        if s and s not in skills:
            skills.append(s)
    return skills

# -------------------------
# Experience & Education
# -------------------------
def extract_experience_years(text: str) -> int:
    if not text:
        return 0

    t = text.lower()

    # 1️⃣ Match decimal years (e.g. 3.5 years)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years|yrs|year)", t)
    if m:
        return int(float(m.group(1)))

    # 2️⃣ Match "X years Y months"
    m = re.search(r"(\d+)\s*years?\s*(\d+)\s*months?", t)
    if m:
        years = int(m.group(1))
        months = int(m.group(2))
        return years + (1 if months >= 6 else 0)

    # 3️⃣ Match "Total Experience: X Years"
    m = re.search(r"total\s+(?:year|years)\s+of\s+experience[:\s]*(\d+)", t)
    if m:
        return int(m.group(1))

    # 4️⃣ Match "Experience: X Years"
    m = re.search(r"experience[:\s]+(\d+)", t)
    if m:
        return int(m.group(1))

    return 0



def education_score(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    t = re.sub(r'\.', ' ', t)
    t = re.sub(r'[,;:/\-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    phd_pattern = r'\b(phd|doctorate)\b'
    master_patterns = r'\b(m\s*sc|msc|mca|m\s*tech|mtech|ms\b|master|mba)\b'
    bachelor_patterns = r'\b(b\s*tech|btech|b\s*e|be\b|beng\b|bachelor|b\s*sc|bsc|bca)\b'
    diploma_pattern = r'\b(diploma|polytechnic|iti)\b'
    hsc_pattern = r'\b(12th|hsc|higher secondary)\b'
    ssc_pattern = r'\b(10th|ssc|secondary school)\b'
    pursuing_pattern = r'\b(pursuing|pursue|in progress|ongoing|currently pursuing|pursuing degree)\b'
    if re.search(phd_pattern, t):
        return 100
    if re.search(master_patterns, t):
        if re.search(pursuing_pattern + r'.{0,60}(' + master_patterns + r')', t) or re.search('(' + master_patterns + r').{0,60}' + pursuing_pattern, t):
            return 75
        return 85
    if re.search(bachelor_patterns, t):
        return 70
    if re.search(diploma_pattern, t):
        return 55
    if re.search(hsc_pattern, t):
        return 40
    if re.search(ssc_pattern, t):
        return 30
    return 0

# -------------------------
# Semantic scoring: SBERT (with TF-IDF fallback)
# -------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
sbert = None
use_sbert = True
try:
    sbert = SentenceTransformer(MODEL_NAME)
except Exception:
    use_sbert = False

def compute_semantic_scores(jd_text: str, resume_texts: list):
    if not resume_texts:
        return [], []
    if use_sbert and sbert is not None:
        try:
            jd_emb = sbert.encode(jd_text, convert_to_tensor=True)
            res_embs = sbert.encode(resume_texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(jd_emb, res_embs).cpu().numpy()[0]
            minv, maxv = float(cosine_scores.min()), float(cosine_scores.max())
            if maxv - minv <= 1e-9:
                norm_scores = np.zeros_like(cosine_scores)
            else:
                norm_scores = 100.0 * (cosine_scores - minv) / (maxv - minv)
            norm_scores = np.round(norm_scores).astype(int)
            return norm_scores.tolist(), cosine_scores.tolist()
        except Exception:
            pass
    # Fallback: TF-IDF + cosine
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=6000)
        docs = [jd_text] + resume_texts
        X = tfidf.fit_transform(docs)
        jd_vec = X[0]
        res_vecs = X[1:]
        cos = cosine_similarity(jd_vec, res_vecs)[0]
        minv, maxv = float(cos.min()), float(cos.max())
        if maxv - minv <= 1e-9:
            norm_scores = np.zeros_like(cos)
        else:
            norm_scores = 100.0 * (cos - minv) / (maxv - minv)
        norm_scores = np.round(norm_scores).astype(int)
        return norm_scores.tolist(), cos.tolist()
    except Exception:
        # graceful fallback zeros
        zeros = [0] * len(resume_texts)
        return zeros, zeros

# -------------------------
# Final score computation
# -------------------------
def compute_final_score(skill_cov_pct: int, semantic_pct: int, exp_years: int, edu_score_val: int):
    exp_contrib = min(exp_years, 10) / 10.0 * 10.0
    edu_contrib = (edu_score_val / 100.0) * 10.0
    weighted = (semantic_pct * 0.55) + (skill_cov_pct * 0.25) + exp_contrib + edu_contrib
    if weighted > 100:
        weighted = 100
    return int(round(weighted))

# -------------------------
# UI: Header
# -------------------------
logo_path = Path("assets/images/logo.png")
col1, col2 = st.columns([1, 10])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=72)
    else:
        st.markdown("<div style='width:72px;height:72px;border-radius:12px;background:#08111a;display:inline-block'></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='app-header'>"
                "<div><h1 class='app-title'>RIGVEDIT — JD ⇄ Resume Matcher</h1>"
                "<div class='app-sub'>Upload a job description and multiple resumes. The app will rank candidates and show breakdowns (Skill / Experience / Education / Semantic).</div>"
                "</div></div>", unsafe_allow_html=True)

st.write("")  # spacing

# -------------------------
# File upload widgets
# -------------------------
jd_file = st.file_uploader("Job Description (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], key="jd")
resume_files = st.file_uploader("Resumes (PDF / DOCX / TXT) — multiple", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resumes")

def show_file_list(files):
    if not files:
        return
    rows = []
    for f in files:
        size_kb = (getattr(f, "size", None) or 0) / 1024
        rows.append(f"<div style='padding:8px 0; color:#d0e7f8; display:flex; align-items:center; gap:12px'>"
                    f"<svg width='18' height='18' viewBox='0 0 24 24' fill='none' style='opacity:0.85'><path d='M7 2h7l5 5v13a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z' stroke='#9fbad0' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'/></svg>"
                    f"<div style='flex:1'>{f.name}</div><div style='color:#9fbad0;font-size:12px'>{size_kb:.0f}KB</div></div>")
    st.markdown("<div style='padding:8px 12px; border-radius:8px; background:rgba(255,255,255,0.01)'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

show_file_list([jd_file] if jd_file else [])
show_file_list(resume_files)

st.write("")  # spacing

# Controls area (right column)
col_left, col_right = st.columns([3, 1])
with col_right:
    st.markdown("<div style='text-align:left'><h2 style='color:#ffffff'>Controls</h2></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#d0e7f8'>Top performers shown: 3</div>", unsafe_allow_html=True)
    run = st.button("Match & Rank")

# -------------------------
# Processing flow
# -------------------------
if run:
    if (not jd_file) or (not resume_files):
        st.warning("Please upload a Job Description and at least one resume.")
    else:
        # Read JD text
        jd_bytes = jd_file.read()
        jd_raw = ""
        jd_ext = jd_file.name.rsplit(".", 1)[-1].lower()
        if jd_ext == "pdf":
            jd_raw = extract_text_from_pdf_bytes(jd_bytes)
        elif jd_ext == "docx":
            jd_raw = extract_text_from_docx_bytes(jd_bytes)
        else:
            try:
                jd_raw = jd_bytes.decode("utf-8", errors="ignore")
            except Exception:
                jd_raw = ""
        jd_text = clean_text(jd_raw)

        # Extract skills
        skills = extract_skills_from_jd(jd_text, topk=40)

        data = []
        resume_texts = []
        for f in resume_files:
            b = f.read()
            ext = f.name.rsplit(".", 1)[-1].lower()
            raw = ""
            if ext == "pdf":
                raw = extract_text_from_pdf_bytes(b)
            elif ext == "docx":
                raw = extract_text_from_docx_bytes(b)
            else:
                try:
                    raw = b.decode("utf-8", errors="ignore")
                except Exception:
                    raw = ""
            text = clean_text(raw)
            resume_texts.append(text)
            # skill coverage (literal substring)
            found = [s for s in skills if s and s in text.lower()]
            coverage = int(round((len(found) / max(len(skills), 1)) * 100))
            years = extract_experience_years(text)
            edu = education_score(text)
            data.append({
                "Resume": f.name,
                "Text": text,
                "Skill_Coverage": coverage,
                "Experience": years,
                "Education": edu
            })

        # Semantic scores
        semantic_norm, semantic_raw = compute_semantic_scores(jd_text, resume_texts)

        # Finalize rows
        for i, row in enumerate(data):
            row["Semantic_pct"] = int(semantic_norm[i]) if semantic_norm else 0
            row["Semantic_raw"] = float(semantic_raw[i]) if semantic_raw else 0.0
            row["Match_Score"] = compute_final_score(row["Skill_Coverage"], row["Semantic_pct"],
                                                    row["Experience"], row["Education"])

        df = pd.DataFrame(data)
        if df.empty:
            st.error("No resume text could be read. Try DOCX or a different PDF.")
        else:
            df = df.sort_values(by="Match_Score", ascending=False).reset_index(drop=True)
            df.insert(0, "No", range(1, len(df) + 1))  # numbering 1..n

            # Present results
            st.markdown("<div class='card'><h2 style='color:#ffffff'>Top Candidates</h2>", unsafe_allow_html=True)
            display_df = df[["No", "Resume", "Skill_Coverage", "Experience", "Education", "Semantic_pct", "Match_Score"]].copy()
            display_df = display_df.rename(columns={
                "Skill_Coverage": "Skill Coverage",
                "Experience": "Experience (yrs)",
                "Education": "Education",
                "Semantic_pct": "Semantic",
                "Match_Score": "Match Score"
            })
            # Format percentages nicely
            display_df["Skill Coverage"] = display_df["Skill Coverage"].astype(str) + "%"
            display_df["Semantic"] = display_df["Semantic"].astype(str) + "%"
            display_df["Match Score"] = display_df["Match Score"].astype(str) + "%"

            st.dataframe(display_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Save CSV bytes for download
            csv_buf = io.BytesIO()
            df_to_csv = df[["Resume", "Skill_Coverage", "Experience", "Education", "Semantic_pct", "Match_Score"]]
            csv_bytes = df_to_csv.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="ranking.csv", mime="text/csv")

            # Top 3 chart (Plotly)
            top3 = df.head(3)
            if not top3.empty:
                top_names = top3["Resume"].tolist()
                top_scores = top3["Match_Score"].tolist()
                # Plotly horizontal bar (reverse order so best appears top)
                colors = ["#ff6b8a", "#4da8ff", "#ffd166"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_scores[::-1],
                    y=top_names[::-1],
                    orientation="h",
                    marker=dict(color=colors[:len(top_scores)][::-1], line=dict(color="rgba(0,0,0,0.15)", width=0)),
                    text=[f"{s}%" for s in top_scores[::-1]],
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{x}%"
                ))
                fig.update_layout(
                    height=360,
                    margin=dict(l=10, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.03)"),
                    yaxis=dict(automargin=True),
                )
                st.markdown("<div style='margin-top:12px'><h2 style='color:#ffffff'>Top 3 Match Scores</h2></div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

        st.balloons()
