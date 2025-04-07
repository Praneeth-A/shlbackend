import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain.schema import Document
# Load JSON
with open(dest_path, "r", encoding="utf-8") as f:
    tests = json.load(f)

# Choose embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast + efficient

# Build corpus & metadata
corpus = []
metadata = []
test_type_mapping = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

for test in tests:
    raw_codes = test.get("assessment_types", [])

    # Expand each valid code using the mapping
    test_type_full = [test_type_mapping.get(code, "") for code in raw_codes if code in test_type_mapping]

    # Join the full names into a string
    test_type_string = ", ".join(test_type_full)
    combined_text = " ".join([
        test.get("name", ""),
        test.get("description", ""),
        "Job Levels: " + test.get("job_levels", ""),
        "Languages: " + test.get("languages", ""),
        "Assessment Length: " + test.get("assessment_length", ""),
        "Remote Testing: " + test.get("remote_testing", ""),
        "Adaptive/IRT: " + test.get("adaptive_irt", ""),f"Test Type: {test_type_string}"
         ])
    test["assessment_types"]=test_type_full
    # Optional chunking if very long
    if len(combined_text) > 1000:
        split_parts = [combined_text[i:i+500] for i in range(0, len(combined_text), 500)]
        for part in split_parts:
            corpus.append(part)
            metadata.append(test)
    else:
        corpus.append(combined_text)
        metadata.append(test)

# Embedding
print(f"🔄 Embedding {len(corpus)} chunks...")
embeddings = model.encode(corpus, show_progress_bar=True)

# FAISS index
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save FAISS index & metadata
faiss.write_index(index, "data/shl_index.faiss")
docstore={}
for i, doc_dict in enumerate(metadata):
    doc_id = str(i)
    # Combine dictionary fields into a flat document string
    # doc_text = " ".join([f"{k}: {v}" for k, v in doc_dict.items()])
    doc = Document(
        page_content="",
        metadata=doc_dict  # full metadata preserved here
    )
    
    docstore[doc_id] = doc
    
with open("data/shl_metadata.pkl", "wb") as f:
    pickle.dump( docstore, f)

print(f"✅ Stored {len(corpus)} chunks in FAISS index.")
