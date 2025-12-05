import gradio as gr
from rapidfuzz import fuzz
import os, pickle

def heuristic_score(jd, resume_text):
    return fuzz.token_set_ratio(jd, resume_text)

model = None
if os.path.exists("model.pkl"):
    try:
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
    except:
        model = None

def run_match(jd, resumes):
    # resumes is a list of strings
    outputs = []
    for i, r in enumerate(resumes):
        if model and hasattr(model, "score"):
            try:
                s = float(model.score(jd, r))
            except:
                s = heuristic_score(jd, r)
        else:
            s = heuristic_score(jd, r)
        outputs.append((f"Resume {i+1}", s))
    outputs_sorted = sorted(outputs, key=lambda x: x[1], reverse=True)
    return outputs_sorted

with gr.Blocks() as demo:
    gr.Markdown("# JD ⇄ Resume Quick Matcher (Gradio)")
    jd = gr.Textbox(lines=8, label="Paste JD here")
    resumes = gr.Textbox(lines=12, label="Paste one or more resumes (separate by `---`)")
    def split_resumes(text):
        parts = [p.strip() for p in text.split('---') if p.strip()]
        return parts
    run_btn = gr.Button("Run matching")
    output = gr.Dataframe(headers=["Resume","Score"], datatype=["str","number"])
    def on_click(jd_text, resumes_text):
        parts = split_resumes(resumes_text)
        res = run_match(jd_text, parts)
        return res
    run_btn.click(on_click, inputs=[jd, resumes], outputs=[output])

if __name__ == "__main__":
    demo.launch()
