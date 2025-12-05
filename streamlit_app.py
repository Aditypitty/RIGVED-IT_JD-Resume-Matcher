import streamlit as st
from rapidfuzz import fuzz
import os
import pickle

st.set_page_config(page_title="JD-Resume Matcher", layout="wide")

st.title("JD ⇄ Resume Quick Matcher")
st.write("A lightweight interface to test JD-vs-resume matching. If you have a trained model file named `model.pkl` in the same folder, the app will try to load it; otherwise a simple heuristic scorer is used.")

uploaded_jd = st.text_area("Paste JD (job description) here", height=200)
uploaded_resumes = st.file_uploader("Upload resumes (multiple .txt files OK)", accept_multiple_files=True)

def heuristic_score(jd, resume):
    # fallback simple matching using rapidfuzz token_set_ratio
    return fuzz.token_set_ratio(jd, resume)

model = None
if os.path.exists("model.pkl"):
    try:
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        st.success("Loaded model.pkl (custom model). App will call model.score(jd, resume) if available.")
    except Exception as e:
        st.warning(f"Found model.pkl but failed to load it: {e}")

if st.button("Run matching"):
    if not uploaded_jd:
        st.error("Please paste a JD.")
    elif not uploaded_resumes:
        st.error("Please upload at least one resume file.")
    else:
        jd_text = uploaded_jd
        results = []
        for f in uploaded_resumes:
            try:
                content = f.read().decode('utf-8', errors='ignore')
            except:
                content = "<binary content - not readable as text>"
            if model and hasattr(model, "score"):
                try:
                    score = float(model.score(jd_text, content))
                except Exception as e:
                    st.warning(f"Model scoring failed for {f.name}: {e}")
                    score = heuristic_score(jd_text, content)
            else:
                score = heuristic_score(jd_text, content)
            results.append({"filename": f.name, "score": score})
        # show results sorted
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        st.subheader("Results (higher = better match)")
        for r in results:
            st.write(f"**{r['filename']}** — Score: {r['score']}")
        st.info("If you have a custom model, add a `model.pkl` with a .score(jd, resume_text) method in the same folder.")
