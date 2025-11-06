import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------
# 1. SCRAPE WEBSITE
# ------------------------------------------------------
def scrape_website(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(separator="\n")
    return text


# ------------------------------------------------------
# 2. CHUNK TEXT
# ------------------------------------------------------
def split_text(text, chunk_size=400):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# ------------------------------------------------------
# 3. VECTOR STORE WITH FAISS
# ------------------------------------------------------
def build_faiss_index(chunks):
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings, embed_model


# ------------------------------------------------------
# 4. LLM LOADING (TinyLlama)
# ------------------------------------------------------
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    return tokenizer, model


# ------------------------------------------------------
# 5. RETRIEVAL
# ------------------------------------------------------
def retrieve(query, index, embed_model, chunks, k=3):
    q_embed = embed_model.encode([query])
    distances, indices = index.search(q_embed, k)
    return [chunks[i] for i in indices[0]]


# ------------------------------------------------------
# 6. LIMIT TOKENS
# ------------------------------------------------------
def truncate_text(text, tokenizer, max_tokens=1800):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)


# ------------------------------------------------------
# 7. LLM ANSWER GENERATION
# ------------------------------------------------------
def generate_answer(context, question, tokenizer, model):
    context = truncate_text(context, tokenizer, max_tokens=1200)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    prompt = truncate_text(prompt, tokenizer, max_tokens=1800)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    if answer.startswith(prompt):
        answer = answer[len(prompt):]

    return answer.strip()


# ======================================================
# ✅ STREAMLIT UI
# ======================================================
st.title("🌐 Website RAG Chatbot (Multi Query)")
st.write("Ask unlimited questions from any website.")

# ---------- Input URL ----------
url = st.text_input("Enter website URL:")
load_btn = st.button("Load Website")

# Session state for FAISS index & chat
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# ---------- Load Website ----------
if load_btn and url:
    st.write("🔄 Scraping website...")
    text = scrape_website(url)

    st.write("🔄 Splitting into chunks...")
    chunks = split_text(text)

    st.write("🔄 Building FAISS vector store...")
    index, embeddings, embed_model = build_faiss_index(chunks)

    # Save in session
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.embed_model = embed_model

    st.write("🔄 Loading TinyLlama LLM...")
    tokenizer, llm = load_llm()
    st.session_state.tokenizer = tokenizer
    st.session_state.llm = llm

    st.success("✅ Website Loaded! You can now ask questions.")


# ---------- Query Section ----------
if st.session_state.index is not None:
    question = st.text_input("Ask your question:")

    if st.button("Ask"):
        top_chunks = retrieve(question,
                              st.session_state.index,
                              st.session_state.embed_model,
                              st.session_state.chunks)

        context = "\n".join(top_chunks)

        answer = generate_answer(context,
                                 question,
                                 st.session_state.tokenizer,
                                 st.session_state.llm)

        st.write("### ✅ Answer:")
        st.write(answer)
