"""
Run: streamlit run app/streamlit_app.py
"""
from dotenv import load_dotenv
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

st.set_page_config(page_title="News Bias Detector",
                   page_icon="🗞️", layout="centered")

st.markdown("""
<style>
    .bias-card { border-radius:12px; padding:20px 24px; margin:12px 0; border-left:6px solid; }
    .bias-Left { background:#fef2f2; border-color:#ef4444; }
    .bias-Center { background:#f0fdf4; border-color:#22c55e; }
    .bias-Right { background:#eff6ff; border-color:#3b82f6; }
    .pill { display:inline-block; padding:4px 14px; border-radius:99px; font-weight:700; }
    .pill-Left { background:#ef4444; color:white; }
    .pill-Center { background:#22c55e; color:white; }
    .pill-Right { background:#3b82f6; color:white; }
</style>
""", unsafe_allow_html=True)
st.markdown("# News Bias Detector")
st.markdown("Classify the political lean of any news article using fine-tuned **DistilBERT** + **Gemini AI** explanation.")
st.divider()

MODEL_DIR = "models.distilbert-bias"


@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    from src.model import BiasClassifier
    return BiasClassifier(MODEL_DIR) if os.path.exists(MODEL_DIR) else None


@st.cache_resource
def load_explainer():
    from src.explainer import BiasExplainer
    try:
        return BiasExplainer()
    except:
        return None


mode = st.radio("input mode", ["Paste text", "Enter URL"], horizontal=True)
text = ""

if mode == "Paste text":
    text = st.text_area("Article text", height=200,
                        placeholder="Place article here...")
else:
    url = st.text_input("Article URL")
    if url:
        with st.spinner("Fetching article..."):
            from src.preprocessing import scrape_article
            text = scrape_article(url) or ""
            if text:
                st.success(f"Fetched {len(text):,} characters")
            else:
                st.error("Could not extract text from that URL.")

explain = st.checkbox("Generate AI explanation", value=True)
go = st.button("Analyze", type="primary", disabled=not text)

if go and text:
    clf = load_classifier()

    if clf is None:
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    with st.spinner("Classifying..."):
        result = clf.predict(text)

    label, conf, scores = result["label"], result["confidence"], result["scores"]

    st.markdown(f"""
    <div class = "bias-card bias-{label}">
        <span class = "pill pill-{label}">{label}</span>
        &nbsp;&nbsp;Confidence: <strong>{conf:.0%}</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Score breakdown**")
    cols = st.columns(3)
    icons = {"Left": "🔴", "Center": "🟢", "Right": "🔵"}
    for col, (lbl, score) in zip(cols, scores.items()):
        col.metric(f"{icons[lbl]} {lbl}", f"{score:.0%}")

    if explain:
        exp = load_explainer()
        if exp:
            with st.spinner("Generating experience..."):
                explanation = exp.explain(text, label, conf)
            st.divider()
            st.markdown(f"### Why **{label}**?")
            st.markdown(explanation)
        else:
            st.warning("Set GEMINI_API_KEY in .env to enable explanation")

st.divider()
st.caption("HuggingFace Transformers · Pytorch · Gemini API · Mlflow · Streamlit")
