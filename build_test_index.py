
import json
from pathlib import Path

import config
import store
from embed import embed_texts

path = Path("chunks/chunks.jsonl")
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

print("Found", len(records), "chunks")
print("First chunk keys:", list(records[0].keys()))
