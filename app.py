import streamlit as st
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_detector import ai_probability


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PlagHealth",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 PlagHealth")
st.subheader("AI-Powered Plagiarism & AI Content Detection")

st.write("Upload a document to analyze plagiarism and AI-generated probability.")

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    classifier = joblib.load("plagiarism_detector_model.pkl")

    faiss_index = faiss.read_index("plagiarism_faiss.index")

    return embedding_model, classifier, faiss_index


embedding_model, classifier, faiss_index = load_models()

# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def chunk_text(text, size=120):

    words = text.split()

    chunks = []

    for i in range(0, len(words), size):

        chunk = " ".join(words[i:i+size])

        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Upload File
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a text file",
    type=["txt"]
)

if uploaded_file:

    text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Document Preview")

    st.write(text[:800])

    # --------------------------------------------------
    # Chunk document
    # --------------------------------------------------

    chunks = chunk_text(text)

    st.write("Total chunks analyzed:", len(chunks))

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    embeddings = embedding_model.encode(chunks)

    scores = []
    plag_chunks = []

    # --------------------------------------------------
    # FAISS similarity search
    # --------------------------------------------------

    for i, emb in enumerate(embeddings):

        D, I = faiss_index.search(np.array([emb]), 1)

        similarity = 1 - D[0][0]

        scores.append(similarity)

        prediction = classifier.predict([[similarity]])[0]

        if prediction == 1:

            plag_chunks.append((chunks[i], similarity))

    # --------------------------------------------------
    # Plagiarism Score
    # --------------------------------------------------

    plagiarism_score = np.mean(scores)

    st.subheader("📊 Plagiarism Score")

    st.metric(
        label="Average Similarity",
        value=round(plagiarism_score, 3)
    )

    # --------------------------------------------------
    # AI Detection
    # --------------------------------------------------

    st.subheader("🤖 AI Generated Content Detection")

    ai_score = ai_probability(text)

    st.metric(
        label="AI Generated Probability",
        value=f"{ai_score}%"
    )

    # --------------------------------------------------
    # Show Plagiarized Chunks
    # --------------------------------------------------

    st.subheader("⚠️ Detected Plagiarized Sections")

    if len(plag_chunks) == 0:

        st.success("No plagiarism detected.")

    else:

        for chunk, score in plag_chunks:

            st.error(f"Similarity Score: {round(score,3)}")

            st.write(chunk)

            st.write("---")