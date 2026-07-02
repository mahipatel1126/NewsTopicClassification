import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ==============================
# Session State
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Page Settings
# ------------------------------
st.set_page_config(
    page_title="News Topic Classification",
    page_icon="📰",
    layout="wide"
)

st.title("📰 AI News Topic Classification")

st.markdown("""
This application classifies a news article into one of the following categories using a fine-tuned **DistilBERT Transformer Model**.

Enter any news article below and click **Predict**.
""")

st.sidebar.markdown("---")
st.sidebar.success("✅ DistilBERT Model Loaded")

# ==============================
# Sidebar
# ==============================
st.sidebar.title("📚 Project Information")

st.sidebar.markdown("""
### 🤖 Model
DistilBERT (Fine-Tuned)

### 📂 Dataset
AG News Dataset

### 📰 Categories
- 🌍 World
- ⚽ Sports
- 💼 Business
- 💻 Sci/Tech

### 🎯 Objective
Predict the topic of a news article using a fine-tuned Transformer model.

""")

st.info(
    "💡 Enter or paste a news article below and click **Predict Category**."
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("best_model")
    model = DistilBertForSequenceClassification.from_pretrained("best_model")
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model.\n\n{e}")
    st.stop()
st.sidebar.info("Transformer: DistilBERT\n\nFramework: Hugging Face")

# Label Mapping
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}
# ------------------------------
# User Input
# ------------------------------
# ==============================
# Example News
# ==============================

st.subheader("📝 Try an Example")

examples = {
    "None": "",
    "🌍 World": "The United Nations held an emergency meeting to discuss the ongoing conflict between two countries.",
    "⚽ Sports": "India defeated Australia by six wickets in the final cricket match.",
    "💼 Business": "The stock market gained 3% after several technology companies reported strong quarterly profits.",
    "💻 Sci/Tech": "Apple launches a new AI chip for future MacBooks and introduces powerful machine learning features."
}

selected = st.selectbox(
    "Choose an example article",
    list(examples.keys())
)

news = st.text_area(
    "Enter News Article",
    value=examples[selected],
    height=220,
    placeholder="Paste or type a news article here..."
)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict Category", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        inputs = tokenizer(
            news,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with st.spinner("🤖 AI is analyzing the news article..."):

            with torch.no_grad():
                outputs = model(**inputs)
        
         

        # Calculate probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Predicted class
        prediction = torch.argmax(probabilities, dim=1).item()

        # Confidence
        confidence = probabilities[0][prediction].item()

        # Save prediction history
        st.session_state.history.append({
            "Article": news[:70] + "..." if len(news) > 70 else news,
            "Prediction": label_map[prediction],
            "Confidence": f"{confidence*100:.2f}%"
        })

        # Display Prediction Result
        st.subheader("✅ Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Prediction\n\n**{label_map[prediction]}**")

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        # Create probability table
        prob_df = pd.DataFrame({
            "Category": list(label_map.values()),
            "Probability": probabilities.numpy()[0]
        })
        st.subheader("📊 Prediction Probabilities")

        st.bar_chart(
            prob_df.set_index("Category")
        )

# ==============================
# Prediction History
# ==============================

if st.session_state.history:
        
    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )       

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )       

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.caption(
    "📰 News Topic Classification using DistilBERT"
)