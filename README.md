# Deploy Package — JD-Resume Quick Matcher

This package contains two lightweight app templates you can use to let HR test JD vs Resume matching quickly:
- `streamlit_app.py` — Streamlit web app (recommended for Streamlit Community Cloud).
- `gradio_app.py` — Gradio web app (recommended for Hugging Face Spaces with Gradio).

## Quick notes
- If you have an actual trained model, place a `model.pkl` (pickle) file in the same folder. The apps will try to call `model.score(jd_text, resume_text)` for scoring. If not present, a simple heuristic (RapidFuzz token set ratio) is used.
- Files are intentionally small and safe — no large model binaries included.

## Deploy to Streamlit Community Cloud (very quick)
1. Create a GitHub repo and push these files to the root (`streamlit_app.py`, `requirements.txt`, `README.md`, optionally `model.pkl`).
2. Go to https://streamlit.io/cloud → Sign in with GitHub → Create app → Choose repo, branch, and `streamlit_app.py` as the main file.
3. Click **Deploy**. After build completes, you'll get a `*.streamlit.app` URL to share.

## Deploy to Hugging Face Spaces (Gradio)
1. Create a new Space on Hugging Face and choose **Gradio** as the SDK.
2. Push `gradio_app.py` and `requirements.txt` to the Space repository.
3. After build, the Space will be available at `https://huggingface.co/spaces/<username>/<space-name>`.

## Run locally (Windows)
1. Create venv:
    python -m venv venv
2. Activate:
    venv\Scripts\activate
3. Install:
    pip install -r requirements.txt
4. Run Streamlit:
    streamlit run streamlit_app.py
5. Or run Gradio:
    python gradio_app.py

## One-click (Windows)
Use `run_streamlit.bat` or `run_gradio.bat` (included) to start the app.

---

If you want, I can:
- Create the GitHub repo with these files (I cannot push to your GitHub without credentials — you must push them).
- Produce a ready email to HR with the URL and instructions once you deploy.
- Customize the UI to mimic your exact matching model inputs (if you paste your current scoring function).
