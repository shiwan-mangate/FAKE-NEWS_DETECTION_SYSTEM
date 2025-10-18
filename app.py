import streamlit as st
import torch
from model_helper import load_model, predict


# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center; color:#2C6DD5;">📰 Fake News Detection System</h1>
    <p style="text-align:center;">Detect misinformation instantly with AI</p>
    <hr style="margin:8px 0;">
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Load Model
# -------------------------------
with st.spinner("Loading system... please wait ⏳"):
    model, tokenizer, device = load_model("robert_fake_news_model.pth")

st.success("System is ready to analyze news content ✅")


# -------------------------------
# Sidebar Info
# -------------------------------
with st.sidebar:
    st.header("About this App")
    st.write(
        """
        This application analyzes news articles, headlines, or short paragraphs
        and determines whether the content is **real** or **fake**.
        """
    )
    st.markdown("---")
    st.caption("👨‍💻 Developed by: **Shiwan Mangate**")


# -------------------------------
# Main Interface
# -------------------------------
st.subheader("🔍 Enter News Text to Analyze")

text_input = st.text_area(
    "Paste a news article, paragraph, or headline below:",
    height=160,
    placeholder="Example: 'Breaking: Scientists discover water on Mars!'"
)


# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🚦 Detect Fake News"):
    user_text = text_input.strip()

    if not user_text:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing content... 🔍"):
            label, confidence = predict(model, tokenizer, user_text, device)
            confidence_pct = confidence * 100 if label == "Fake News" else (1 - confidence) * 100

        # Result Display
        color = "#E63946" if label == "Fake News" else "#2A9D8F"
        st.markdown(
            f"<h3 style='text-align:center; color:{color};'>{label}</h3>",
            unsafe_allow_html=True
        )

        st.progress(int(confidence_pct))
        st.info(f"Confidence level: **{confidence_pct:.2f}%**")

        if label == "Fake News":
            st.warning("This article seems unreliable or misleading.")
        else:
            st.success("This news appears authentic and trustworthy.")
