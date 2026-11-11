import json
from pathlib import Path
import streamlit as st

st.title("Chunk Data Test")

path = Path(__file__).parent / "chunks" / "chunks.jsonl"

if path.exists():
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    st.success(f"Found {len(records)} chunks")
    if records:
        st.write("First chunk keys:", list(records[0].keys()))
else:
    st.error("Chunk file not found")
