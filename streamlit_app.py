# streamlit_app.py
import os
import io
import re
import math
import pickle
import base64
import pandas as pd
import streamlit as st

# optional imports (we attempt to import and degrade gracefully)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

# page config
st.set_page_config(page_title="RIGVEDIT — JD ⇄ Resume Matcher", layout="wide")

# ---- Styling: dark, card-like UI, larger fonts and contrast for labels ----
st.markdown(
    """
    <style>
    :root{
      --bg1: #061023;
      --bg2: #071427;
      --card: rgba(255,255,255,0.02);
      --muted: rgba(255,255,255,0.62);
      --muted-weak: rgba(255,255,255,0.44);
      --text: #e6eef6;
      --accent: #00e6b8;
      --accent2: #2bb3ff;
    }
    .stApp { background: linear-gradient(180deg,var(--bg1),var(--bg2)); color:var(--text); font-family: Inter, system-ui, -apple-system, "Helvetica Neue", Arial; }
    .big-title { font-size:44px !important; font-weight:800; color:var(--text) !important; margin:0 0 6px 0; }
    .subtitle { color:var(--muted) !important; font-size:16px !important; margin:0 0 20px 0; }

    /* file uploader boxes */
    .stFileUploader, .upload-box {
      background: var(--card) !important;
      border-radius: 12px !important;
      padding: 14px !important;
      box-shadow: 0 8px 30px rgba(0,0,0,0.45);
      color: var(--muted) !important;
    }

    /* labels */
    label, .stMarkdown p, .stText, .stExpanderHeader {
      color: var(--muted) !important;
      font-size:14px !important;
    }

    /* buttons */
    .stButton>button {
      background: linear-gradient(90deg,var(--accent),var(--accent2)) !important;
      color: #08111a !important;
      border: none !important;
      padding: 10px 16px !important;
      border-radius: 10px !important;
      font-weight:700;
      box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }

    /* card where results are shown */
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
      border-radius: 12px !important;
      padding: 20px !important;
      color: var(--text) !important;
      box-shadow: 0 10px 30px rgba(2,6,23,0.6);
    }

    /* logo size */
    .logo-img { width:72px; height:72px; border-radius:12px; object-fit:cover; box-shadow:0 6px 18px rgba(0,0,0,0.6); margin-right:14px; }

    /* table headings */
    table, th, td { color:var(--text) !important; }
    th { color:var(--muted) !important; font-size:13px !important; }

    /* Extra padding for the main container */
    .css-1d391kg { padding-left:40px !important; padding-right:40px !important; padding-top:20px !important; }
    @media (max-width: 900px) {
      .big-title { font-size:32px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header: logo + title ----
header_col, spacer = st.columns([0.085, 0.915])
with header_col:
    # logo path - ensure you added assets/images/logo.png in repo
    logo_path = "assets/images/logo.png"
    if os.path.exists(logo_path):
        st.markdown(f'<img src="{logo_path}" class="logo-img">', unsafe_allow_html=True)
    else:
        # fallback small inline emoji if logo missing
        st.markdown('<div style="width:72px;height:72px;border-radius:12px;background:#0b2a33;"></div>', unsafe_allow_html=True)

with spacer:
    st.markdown('<div class="big-title">RIGVEDIT — JD ⇄ Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a job description and multiple resumes. The app will rank candidates and show breakdowns (Skill / Experience / Education / Semantic).</div>', unsafe_allow_html=True)

# ---- Upload area (left) and controls (right) ----
col_left, col_right = st.columns([3,1])

with col_left:
    jd_file = st.file_uploader("Job Description (PDF / DOCX / TXT)", type=['pdf','docx','txt'])
    resumes = st.file_uploader("Resumes (PDF / DOCX / TXT) — multiple", type=['pdf','docx','txt'], accept_multiple_files=True)

with col_right:
    st.markdown("### Controls")
    st.markdown("**Top performers shown: 3**")
    run_btn = st.button("Match & Rank")

# -------------------------
# Helper: extract text from files
# -------------------------
def extract_text_pdf_bytes(b):
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                return " ".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            return ""
    # fallback
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_docx_bytes(b):
    if docx:
        try:
            doc = docx.Document(io.BytesIO(b))
            return " ".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def file_bytes_to_text(f):
    name = f.name.lower()
    b = f.read()
    if name.endswith(".pdf"):
        return extract_text_pdf_bytes(b)
    if name.endswith(".docx"):
        return extract_text_docx_bytes(b)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return str(b)

def clean_text(t):
    if not t:
        return ""
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# simple heuristics
COMMON_DEGREES = ['bachelor','b.tech','b.e','b.sc','b.com','m.tech','m.sc','mca','mba','master','phd','doctor']
def extract_experience_years(text):
    m = re.search(r'(\d{1,2})\+?\s*(?:years|yrs)\b', text.lower())
    if m:
        return int(m.group(1))
    yrs = re.findall(r'(19|20)\d{2}', text)
    if len(yrs) >= 2:
        try:
            return abs(int(yrs[-1]) - int(yrs[0]))
        except:
            pass
    return 0

def education_score(text):
    t = (text or "").lower()
    score = 0
    for d in COMMON_DEGREES:
        if d in t:
            score = 70
            break
    if 'master' in t or 'm.tech' in t or 'mba' in t:
        score = max(score, 90)
    if 'phd' in t or 'doctor' in t:
        score = 100
    return score

def skill_coverage_score(jd_text, resume_text):
    if not jd_text or not resume_text:
        return 0.0
    if fuzz:
        return fuzz.token_set_ratio(jd_text, resume_text)  # 0-100
    # fallback simple overlap
    jd_words = set(re.findall(r'\w{3,}', jd_text.lower()))
    r_words = set(re.findall(r'\w{3,}', resume_text.lower()))
    if not jd_words:
        return 0.0
    return float(len(jd_words & r_words) / len(jd_words) * 100.0)

def semantic_similarity_score(jd_text, resume_text, vect=None):
    if not jd_text.strip() or not resume_text.strip():
        return 0.0
    if TfidfVectorizer is None:
        return 0.0
    docs = [jd_text, resume_text]
    if vect is None:
        vect = TfidfVectorizer(stop_words='english', max_features=3000)
        X = vect.fit_transform(docs)
    else:
        X = vect.transform(docs)
    try:
        sim = cosine_similarity(X[0], X[1])[0][0]
    except Exception:
        sim = 0.0
    return float(sim) * 100.0

def compute_final_score(skill_cov_pct, semantic_pct, exp_years, edu_score):
    # weighted scoring (semantic heavier)
    exp_contrib = min(exp_years, 10) / 10.0 * 10.0
    edu_contrib = (edu_score / 100.0) * 10.0
    weighted = (semantic_pct * 0.55) + (skill_cov_pct * 0.25) + exp_contrib + edu_contrib
    weighted = min(weighted, 100)
    return int(round(weighted))

# -------------------------
# Run matching
# -------------------------
if run_btn:
    if not jd_file:
        st.error("Please upload a job description file.")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Processing files..."):
            jd_text = file_bytes_to_text(jd_file)
            jd_text = clean_text(jd_text)

            names = []
            resume_texts = []
            for f in resumes:
                names.append(f.name)
                resume_texts.append(clean_text(file_bytes_to_text(f)))

            # build vectorizer on JD+resumes for TF-IDF semantic fallback
            vect = None
            if TfidfVectorizer:
                try:
                    vect = TfidfVectorizer(stop_words='english', max_features=3000)
                    vect.fit([jd_text] + resume_texts)
                except Exception:
                    vect = None

            rows = []
            for name, text in zip(names, resume_texts):
                skill_cov = skill_coverage_score(jd_text, text)
                exp = extract_experience_years(text)
                edu = education_score(text)
                sem = semantic_similarity_score(jd_text, text, vect=vect)
                match_score = compute_final_score(skill_cov, sem, exp, edu)
                rows.append({
                    "Resume": name,
                    "Skill Coverage": f"{int(round(skill_cov))}%",
                    "Experience (yrs)": int(exp),
                    "Education": int(edu),
                    "Semantic": f"{int(round(sem))}%",
                    "Match Score": match_score
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("Match Score", ascending=False).reset_index(drop=True)

            # display results in card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            left, right = st.columns([2,1])
            with left:
                st.subheader("Top Candidates")
                # show only top 3 as design
                st.dataframe(df.head(5), use_container_width=True, height=340)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name="match_results.csv", mime="text/csv")

            with right:
                st.subheader("Top 3 Match Scores")
                top3 = df.head(3)
                if not top3.empty:
                    # horizontal bars via plotly (if available) else streamlit bar_chart
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=top3["Match Score"].astype(int).tolist()[::-1],
                            y=top3["Resume"].tolist()[::-1],
                            orientation='h',
                            marker=dict(color=['#ff7f9e','#2fa4ff','#ffd166'])
                        ))
                        fig.update_layout(margin=dict(l=20,r=10,b=10,t=10), xaxis=dict(range=[0,100], showgrid=True))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        chart_df = pd.DataFrame({'Candidate': top3['Resume'], 'Score': top3['Match Score'].astype(float)}).set_index('Candidate')
                        st.bar_chart(chart_df)
            st.markdown('</div>', unsafe_allow_html=True)
            st.success("Done — scroll to see results below.")
