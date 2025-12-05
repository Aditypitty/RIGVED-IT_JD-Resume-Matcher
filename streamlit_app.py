# streamlit_app.py
import streamlit as st
import pdfplumber
import docx
import re
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import plotly.graph_objects as go

st.set_page_config(page_title="RIGVEDIT — JD Resume Matcher", layout="wide", initial_sidebar_state="auto")

# ---- Styling to mimic the dark, card-style UI ----
st.markdown(
    """
    <style>
    :root { --bg: #061022; --card:#081826; --muted:#9fb0c5; --accent:#00c2a8; --text:#e6eef6; }
    .stApp { background: linear-gradient(180deg,#061022,#07182a); color: var(--text); }
    .card { background: var(--card); padding: 24px; border-radius: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
    .big-title { font-size:34px; font-weight:700; color:var(--text); margin-bottom:6px; }
    .muted { color: var(--muted); font-size:14px; margin-bottom:18px; }
    .download-btn { background: linear-gradient(90deg,#06d3a0,#0ea6ff); color:#fff; padding:10px 18px; border-radius:10px; text-decoration:none; }
    .table-header { color: #bcd3e0; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">RIGVEDIT — JD ⇄ Resume Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Upload a job description and multiple resumes. The app will rank candidates and show breakdowns (Skill / Experience / Education / Semantic).</div>', unsafe_allow_html=True)

# Layout: left = inputs + results table, right = controls + chart
left, right = st.columns([2.5,1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Job Description & Resumes")
    jd_file = st.file_uploader("Job Description (PDF / DOCX / TXT)", type=['pdf','docx','txt'])
    resumes = st.file_uploader("Resumes (PDF / DOCX / TXT) — multiple", type=['pdf','docx','txt'], accept_multiple_files=True)
    st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.04)'/>", unsafe_allow_html=True)

    # placeholder for results table (rendered after matching)
    results_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Controls")
    topk = st.number_input("Top candidates to show", min_value=1, max_value=20, value=5)
    run_btn = st.button("Match & Rank")
    st.markdown("</div>", unsafe_allow_html=True)

# --- helper functions for text extraction and scoring ---
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""
    return text

def extract_text_from_docx(file_bytes):
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        text = ""
    return text

def file_to_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    b = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)
    if name.endswith(".docx"):
        return extract_text_from_docx(b)
    try:
        return b.decode('utf-8', errors='ignore')
    except:
        return str(b)

COMMON_DEGREES = ['bachelor','b.tech','b.e','b.sc','b.com','bca','m.tech','m.sc','mca','mba','master','phd','doctor']

def extract_experience_years(text):
    text_l = text.lower()
    # direct pattern: "5 years", "5+ years", "5 yrs"
    m = re.search(r'(\d{1,2})\+?\s*(?:years|yrs|year)\b', text_l)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    # years range heuristic e.g., 2015 - 2020
    years = re.findall(r'((?:19|20)\d{2})', text_l)
    if len(years) >= 2:
        try:
            return abs(int(years[-1]) - int(years[0]))
        except:
            pass
    return 0

def education_score(text):
    t = text.lower()
    score = 0
    for d in COMMON_DEGREES:
        if d in t:
            score = 70
            break
    if 'master' in t or 'm.tech' in t or 'mba' in t or 'm.sc' in t or 'mca' in t:
        score = max(score, 90)
    if 'phd' in t or 'doctor' in t:
        score = 100
    return int(score)

def skill_coverage_score(jd_text, resume_text):
    # rapidfuzz token_set_ratio (0-100)
    try:
        return int(fuzz.token_set_ratio(jd_text, resume_text))
    except:
        return 0

def semantic_similarity_score(jd_text, resume_text, vect=None):
    if not jd_text.strip() or not resume_text.strip():
        return 0.0
    docs = [jd_text, resume_text]
    if vect is None:
        vect = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vect.fit_transform(docs)
    else:
        X = vect.transform(docs)
    score = cosine_similarity(X[0], X[1])[0][0]
    return float(score) * 100.0

# run matching when button pressed
if run_btn:
    if not jd_file:
        st.error("Please upload a Job Description file.")
    elif not resumes:
        st.error("Please upload at least one resume file.")
    else:
        with st.spinner("Processing and scoring resumes..."):
            jd_text = file_to_text(jd_file)
            candidate_texts = []
            candidate_names = []
            for f in resumes:
                txt = file_to_text(f)
                candidate_texts.append(txt)
                candidate_names.append(f.name)

            # pre-fit vectorizer on JD + resumes for semantic similarity
            vect = TfidfVectorizer(stop_words='english', max_features=3000)
            try:
                vect.fit([jd_text] + candidate_texts)
            except:
                pass

            rows = []
            for name, text in zip(candidate_names, candidate_texts):
                skill = skill_coverage_score(jd_text, text)  # 0-100
                exp = extract_experience_years(text)
                edu = education_score(text)  # 0-100-ish
                sem = round(semantic_similarity_score(jd_text, text, vect=vect), 2)
                # match score: tune weights to match your original model visuals
                match_score = round(0.45 * skill + 0.15 * min(exp * 5, 20) + 0.2 * edu + 0.2 * sem, 2)
                rows.append({
                    "Resume": name,
                    "Skill Coverage": f"{int(skill)}%",
                    "Experience (yrs)": int(exp),
                    "Education": int(edu),
                    "Semantic": f"{int(sem)}%",
                    "Match Score": match_score
                })

            df = pd.DataFrame(rows).sort_values("Match Score", ascending=False).reset_index(drop=True)

        # render the richer UI (table + chart + download)
        results_placeholder.empty()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns([2.2,1])
        with cols[0]:
            st.subheader("Top Candidates")
            # use st.dataframe for scrollable table
            styled = df.head(topk)
            st.dataframe(styled, use_container_width=True, height=360)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="match_results.csv", mime="text/csv")
        with cols[1]:
            st.subheader("Top 3 Match Scores")
            top3 = df.head(3)
            if not top3.empty:
                # Plotly horizontal bars with nicer colors and rounded style
                fig = go.Figure()
                colors = ['#ff6b8a', '#3da7ff', '#ffcc56']
                fig.add_trace(go.Bar(
                    y=top3['Resume'],
                    x=top3['Match Score'],
                    orientation='h',
                    marker=dict(color=colors[:len(top3)], line=dict(color='rgba(0,0,0,0.2)', width=0)),
                    text=top3['Match Score'],
                    textposition='inside'
                ))
                fig.update_layout(
                    xaxis=dict(range=[0, max(100, top3['Match Score'].max()*1.1)]),
                    template='plotly_dark',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("Done — scroll to see results above.")
