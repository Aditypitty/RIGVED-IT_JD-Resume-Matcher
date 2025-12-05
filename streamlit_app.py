# streamlit_app.py
import streamlit as st
import io, os, re, tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# text extraction
import pdfplumber
import docx

# keyword extraction + fuzzy
import yake
from rapidfuzz import fuzz

# vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional SBERT
SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# Page config
st.set_page_config(page_title="RIGVEDIT — JD Resume Matcher", layout="wide")

# Inject your CSS (from Style.css) so Streamlit looks like your original
st.markdown("""
<style>
:root{
  --bg:#0f1724;
  --card:#0b1220;
  --muted:#9aa6b2;
  --accent:#00b894;
  --accent2:#0984e3;
  --glass: rgba(255,255,255,0.03);
  --text:#e6eef6;
  --surface:#08111a;
}
body { background: linear-gradient(180deg,#061023 0%, #071427 100%); color:var(--text); }
.big-title{ font-size:34px; font-weight:700; margin-bottom:6px; color:var(--text); }
.muted { color: var(--muted); margin-bottom:18px; display:block; }
.card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); padding:20px; border-radius:12px; box-shadow: 0 6px 30px rgba(2,6,23,0.6); margin-bottom:20px; }
.header{ display:flex; gap:14px; align-items:center; margin-bottom:20px; }
.logo{ width:64px; height:64px; object-fit:contain; border-radius:10px; background:linear-gradient(135deg,var(--accent),var(--accent2)); padding:8px; box-shadow: 0 6px 20px rgba(0,0,0,0.5); }
.result-table th, .result-table td{ color:var(--text); }
.download-btn { background: linear-gradient(90deg,#06d3a0,#0ea6ff); color:#08111a; border-radius:10px; padding:8px 12px; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

# Header with logo (logo path: assets/images/logo.png)
logo_path = "assets/images/logo.png"
colL, colR = st.columns([1,9])
with colL:
    if Path(logo_path).exists():
        st.image(logo_path, width=72)
    else:
        # try raw GitHub URL (public repo)
        try:
            st.image("https://raw.githubusercontent.com/Aditypitty/RIGVED-IT_JD-Resume-Matcher/main/assets/images/logo.png", width=72)
        except:
            pass
with colR:
    st.markdown('<div class="big-title">RIGVEDIT — JD ⇄ Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload a job description and multiple resumes. The app will rank candidates and show breakdowns (Skill / Experience / Education / Semantic).</div>', unsafe_allow_html=True)

# layout
left, right = st.columns([2.5,1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Job Description & Resumes")
    jd_file = st.file_uploader("Job Description (PDF / DOCX / TXT)", type=['pdf','docx','txt'])
    resumes = st.file_uploader("Resumes (PDF / DOCX / TXT) — multiple", type=['pdf','docx','txt'], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Controls")
    topk = st.number_input("Top candidates to show", min_value=1, max_value=20, value=5)
    run_btn = st.button("Match & Rank")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Helpers (adapted from your Flask code) ----------
def extract_text_pdf_bytes(b: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    text += " " + txt
    except Exception:
        text = ""
    return text or ""

def extract_text_docx_bytes(b: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(io.BytesIO(b))
        for p in doc.paragraphs:
            text += " " + p.text
    except Exception:
        text = ""
    return text or ""

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\n"," ").replace("\r"," ")
    t = re.sub(r'\s+',' ', t)
    return t.strip()

# skills extraction (YAKE + noun chunks via yake only to mimic)
def extract_skills_from_jd(jd_text: str, topk: int = 40):
    if not jd_text:
        return []
    kw = yake.KeywordExtractor(top=topk, stopwords=None)
    yake_res = kw.extract_keywords(jd_text)
    yake_kw = [k for k, s in yake_res]
    skills = []
    for s in yake_kw:
        s = s.lower().strip()
        if s and s not in skills:
            skills.append(s)
    return skills

# experience & education (use your functions)
COMMON_DEGREES = ['bachelor','b.tech','b.e','b.sc','b.com','bca','m.tech','m.sc','mca','mba','master','phd','doctor']

def extract_experience_years(text: str) -> int:
    if not text:
        return 0
    matches = re.findall(r"(\d{1,2})\s*(?:\+)?\s*(?:years|yrs|yrs\.|year)\b", text.lower())
    if matches:
        years = max(int(m) for m in matches)
        return years
    m = re.search(r"experience[:\s]+(\d{1,2})", text.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    years = re.findall(r'((?:19|20)\d{2})', text)
    if len(years) >= 2:
        try:
            return abs(int(years[-1]) - int(years[0]))
        except:
            pass
    return 0

def education_score(text: str) -> int:
    t = text.lower()
    t = re.sub(r'\.', ' ', t)
    t = re.sub(r'[,;:/\\-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if re.search(r'\b(phd|doctorate)\b', t):
        return 100
    if re.search(r'\b(m\s*sc|msc|mca|m\s*tech|mtech|ms\b|master|mba)\b', t):
        if re.search(r'\b(pursuing|pursue|ongoing|currently pursuing)\b', t):
            return 75
        return 85
    if re.search(r'\b(b\s*tech|btech|b\s*e|be\b|beng\b|bachelor|b\s*sc|bsc|bca)\b', t):
        return 70
    if re.search(r'\b(diploma|polytechnic|iti)\b', t):
        return 55
    if re.search(r'\b(12th|hsc|higher secondary)\b', t):
        return 40
    if re.search(r'\b(10th|ssc|secondary school)\b', t):
        return 30
    return 0

# semantic scoring: try SBERT if available, else TF-IDF fallback
def compute_semantic_scores(jd_text: str, resume_texts: list):
    if not resume_texts:
        return [], []
    # SBERT path
    if SBERT_AVAILABLE:
        try:
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            jd_emb = model.encode(jd_text, convert_to_tensor=True)
            res_embs = model.encode(resume_texts, convert_to_tensor=True)
            cos = util.cos_sim(jd_emb, res_embs).cpu().numpy()[0]
            minv, maxv = float(cos.min()), float(cos.max())
            if maxv - minv <= 1e-9:
                norm = np.zeros_like(cos)
            else:
                norm = 100.0 * (cos - minv) / (maxv - minv)
            return norm.astype(int).tolist(), cos.tolist()
        except Exception:
            pass
    # TF-IDF fallback
    vect = TfidfVectorizer(stop_words='english', max_features=3000)
    try:
        X = vect.fit_transform([jd_text] + resume_texts)
        jd_vec = X[0]
        res_vecs = X[1:]
        cos = cosine_similarity(jd_vec, res_vecs)[0]
        minv, maxv = float(cos.min()), float(cos.max())
        if maxv - minv <= 1e-9:
            norm = np.zeros_like(cos)
        else:
            norm = 100.0 * (cos - minv) / (maxv - minv)
        return norm.astype(int).tolist(), cos.tolist()
    except Exception:
        return [0]*len(resume_texts), [0.0]*len(resume_texts)

def compute_final_score(skill_cov_pct:int, semantic_pct:int, exp_years:int, edu_score_val:int):
    exp_contrib = min(exp_years, 10) / 10.0 * 10.0
    edu_contrib = (edu_score_val / 100.0) * 10.0
    weighted = (semantic_pct * 0.55) + (skill_cov_pct * 0.25) + exp_contrib + edu_contrib
    if weighted > 100:
        weighted = 100
    return int(round(weighted))

# ---------- Run matching ----------
if run_btn:
    if not jd_file:
        st.error("Please upload a Job Description file.")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Processing files..."):
            # extract JD text
            jd_bytes = jd_file.read()
            if jd_file.name.lower().endswith(".pdf"):
                jd_raw = extract_text_pdf_bytes(jd_bytes)
            elif jd_file.name.lower().endswith(".docx"):
                jd_raw = extract_text_docx_bytes(jd_bytes)
            else:
                try:
                    jd_raw = jd_bytes.decode('utf-8', errors='ignore')
                except:
                    jd_raw = ""
            jd_text = clean_text(jd_raw)

            # extract skills
            skills = extract_skills_from_jd(jd_text, topk=40)

            candidate_texts = []
            names = []
            data = []
            for f in resumes:
                names.append(f.name)
                fb = f.read()
                if f.name.lower().endswith(".pdf"):
                    txt = extract_text_pdf_bytes(fb)
                elif f.name.lower().endswith(".docx"):
                    txt = extract_text_docx_bytes(fb)
                else:
                    try:
                        txt = fb.decode('utf-8', errors='ignore')
                    except:
                        txt = ""
                txt_clean = clean_text(txt)
                candidate_texts.append(txt_clean)

                # skill coverage: substring check against extracted skills
                found = [s for s in skills if s and s in txt_clean]
                coverage = int(round((len(found)/max(len(skills),1))*100))
                years = extract_experience_years(txt_clean)
                edu = education_score(txt_clean)
                data.append({
                    "Resume": f.name,
                    "Text": txt_clean,
                    "Skill_Coverage": coverage,
                    "Experience": years,
                    "Education": edu
                })

            # semantic
            semantic_norm, semantic_raw = compute_semantic_scores(jd_text, candidate_texts)

            # finalize rows
            for i, row in enumerate(data):
                spct = int(semantic_norm[i]) if semantic_norm else 0
                row["Semantic_pct"] = spct
                row["Semantic_raw"] = float(semantic_raw[i]) if semantic_raw else 0.0
                row["Match_Score"] = compute_final_score(row["Skill_Coverage"], spct, row["Experience"], row["Education"])
                row["Skill_Coverage_str"] = f"{row['Skill_Coverage']}%"
                row["Semantic_str"] = f"{row['Semantic_pct']}%"
                row["Match_Score_str"] = f"{row['Match_Score']}%"

            df = pd.DataFrame(data).sort_values(by="Match_Score", ascending=False).reset_index(drop=True)

        # show results in card (table left, chart right)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns([2,1])
        with cols[0]:
            st.subheader("Top Candidates")
            # show table similar to your HTML table
            display_df = df[["Resume","Skill_Coverage_str","Experience","Education","Semantic_str","Match_Score"]].copy()
            display_df.columns = ["Resume","Skill Coverage","Experience (yrs)","Education","Semantic","Match Score"]
            st.table(display_df.head(topk).replace({np.nan:""}))
            # download CSV
            csv = df[["Resume","Skill_Coverage","Experience","Education","Semantic_pct","Match_Score"]].to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="ranking.csv", mime="text/csv")
        with cols[1]:
            st.subheader("Top 3 Match Scores")
            top3 = df.head(3)
            if not top3.empty:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=top3["Match_Score"],
                    y=top3["Resume"],
                    orientation='h',
                    marker=dict(color=['rgba(255,99,132,0.95)','rgba(54,162,235,0.95)','rgba(255,205,86,0.95)']),
                    text=top3["Match_Score"],
                    textposition='inside'
                ))
                fig.update_layout(template='plotly_dark', xaxis=dict(range=[0,100], tick0=0), margin=dict(l=10,r=10,t=10,b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("Done — scroll to see results above.")
