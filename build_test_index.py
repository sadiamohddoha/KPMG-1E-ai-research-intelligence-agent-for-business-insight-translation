
import json
from pathlib import Path

import streamlit as st

import config
import store
from embed import embed_texts

st.title("Build Test Search Index")

@st.cache_resource
def build_index():
    path = Path(__file__).parent / "chunks" / "chunks.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    collection = store.get_collection()

    ids = [str(r["chunk_id"]) for r in records]
    texts = [r["chunk_text"] for r in records]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
    )

    return len(records)

if st.button("Build search index"):
    with st.spinner("Building index..."):
        count = build_index()
    st.success(f"Indexed {count} chunks!")

