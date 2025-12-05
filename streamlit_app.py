# streamlit_app.py
import streamlit as st
import pdfplumber
import docx
import re
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

st.set_page_config(page_title="RIGVED JD–Resume Matcher", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071029,#0b1b2b); color: #e6eef6; }
    .big-title { font-size:36px; font-weight:700; color:#fff; }
    .card { background:#081826; padding:22px; border-radius:12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">RIGVEDIT — JD ⇄ Resume Matcher</div>', unsafe_allow_html=True)
st.write("Upload a job description and multiple resumes. The app will rank candidates and give breakdowns.")

col1, col2 = st.columns([2,1])

with col1:
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=['pdf','docx','txt'])
    resumes = st.file_uploader("Resumes (PDF/DOCX/TXT) — multiple", type=['pdf','docx','txt'], accept_multiple_files=True)

with col2:
    st.markdown("### Controls")
    topk = st.number_input("Top candidates to show", min_value=1, max_value=20, value=5)
    run_btn = st.button("Match & Rank")

# helper functions
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
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

def file_to_text(f):
    name = f.name.lower()
    b = f.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)
    if name.endswith(".docx"):
        return extract_text_from_docx(b)
    # assume text
    try:
        return b.decode('utf-8', errors='ignore')
    except:
        return str(b)

# simple heuristics for skill coverage, experience, education
COMMON_DEGREES = ['bachelor','b.tech','b.e','b.sc','b.com','m.tech','m.sc','mca','mba','master','phd','doctor']
def extract_experience_years(text):
    # look for patterns like "5 years", "5+ years", "five years"
    m = re.search(r'(\d{1,2})\+?\s*(?:years|yrs)\b', text.lower())
    if m:
        return int(m.group(1))
    # fallback: try ranges "2015 - 2020" -> assume 5 years
    yrs = re.findall(r'(19|20)\d{2}', text)
    if len(yrs) >= 2:
        try:
            return abs(int(yrs[-1]) - int(yrs[0]))
        except:
            pass
    return 0

def education_score(text):
    t = text.lower()
    score = 0
    for d in COMMON_DEGREES:
        if d in t:
            score = 70  # base score for matching degree keyword
            break
    # try masters/phd boost
    if 'master' in t or 'm.tech' in t or 'mba' in t or 'm.sc' in t:
        score = max(score, 90)
    if 'phd' in t or 'doctor' in t:
        score = 100
    return score

def skill_coverage_score(jd_text, resume_text):
    # simple token overlap using rapidfuzz token_set_ratio scaled
    return fuzz.token_set_ratio(jd_text, resume_text) / 100.0 * 100  # as percentage

def semantic_similarity_score(jd_text, resume_text, vect=None):
    # fallback: TFIDF cosine
    if not jd_text.strip() or not resume_text.strip():
        return 0.0
    docs = [jd_text, resume_text]
    if vect is None:
        vect = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vect.fit_transform(docs)
    else:
        X = vect.transform(docs)
    score = cosine_similarity(X[0], X[1])[0][0]
    return float(score) * 100.0  # as percentage

if run_btn:
    if not jd_file:
        st.error("Please upload a JD file.")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Processing files..."):
            jd_text = file_to_text(jd_file)
            resume_texts = []
            names = []
            for f in resumes:
                txt = file_to_text(f)
                resume_texts.append(txt)
                names.append(f.name)

            # vectorizer trained on JD + resumes for semantic TF-IDF
            vect = TfidfVectorizer(stop_words='english', max_features=3000)
            try:
                vect.fit([jd_text] + resume_texts)
            except:
                pass

            rows = []
            for name, text in zip(names, resume_texts):
                skill_cov = round(skill_coverage_score(jd_text, text), 2)
                exp = extract_experience_years(text)
                edu = education_score(text)
                sem = round(semantic_similarity_score(jd_text, text, vect=vect), 2)
                # Combine scores (weights can be tuned)
                match_score = round(0.45*skill_cov + 0.15*min(exp*5,20) + 0.2*edu + 0.2*sem, 2)
                rows.append({
                    "Resume": name,
                    "Skill Coverage": f"{int(skill_cov)}%",
                    "Experience (yrs)": exp,
                    "Education": int(edu),
                    "Semantic": f"{int(sem)}%",
                    "Match Score": match_score
                })

            df = pd.DataFrame(rows)
            df = df.sort_values("Match Score", ascending=False).reset_index(drop=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            left, right = st.columns([2,1])
            with left:
                st.subheader("Top Candidates")
                st.dataframe(df.head(int(topk)), use_container_width=True, height=360)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name="match_results.csv", mime="text/csv")

            with right:
                st.subheader("Top 3 Match Scores")
                top3 = df.head(3)
                if not top3.empty:
                    chart_df = pd.DataFrame({
                        'Candidate': top3['Resume'],
                        'Score': top3['Match Score'].astype(float)
                    }).set_index('Candidate')
                    st.bar_chart(chart_df)

            st.markdown("</div>", unsafe_allow_html=True)
            st.success("Done — scroll to see results below.")
