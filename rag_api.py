# rag_api.py
import os
import faiss
import pickle
# import numpy as np
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
# Load FAISS index and docstore
try:
    with open("demoData/shl_demo_metadata.pkl", "rb") as f:
        docstore = pickle.load(f)
    # with open("embeddings/index_to_docstore_id.pkl", "rb") as f:
    #     index_to_docstore_id = pickle.load(f)
    index = faiss.read_index("demoData/shl_demo_index.faiss")

    # LangChain FAISS setup
    # vectorstore = FAISS(
    #     embedding_function=None,
    #     index=index,
    #     docstore=InMemoryDocstore(docstore),
    #     index_to_docstore_id={i: str(i) for i in range(index.ntotal)}
    # ,
    # )
        # Create a new FAISS index with just the first 30 vectors
    new_index = faiss.IndexFlatL2(index.d)  # assumes L2 type
    xb = faiss.vector_to_array(index.xb).reshape(index.ntotal, -1)[:100]
    new_index.add(xb)

    # Create a smaller docstore with only 30 entries
    limited_docstore = {str(i): docstore[str(i)] for i in range(100)}

    vectorstore = FAISS(
        embedding_function=None,
        index=new_index,
        docstore=InMemoryDocstore(limited_docstore),
        index_to_docstore_id={i: str(i) for i in range(100)}
    )

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Gemini setup
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro-002")

    
    
except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        # return jsonify({"name":"kkkk","new":"yes"})
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        query_embedding = embedding_model.embed_query(query)

        # Hybrid FAISS approach
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(query_embedding, k=20)

        # Filter based on scores, guarantee at least 1
        threshold = 0.4  # adjust as needed
        filtered_docs = [doc for doc, score in docs_and_scores if score < threshold]
        if not filtered_docs:
            filtered_docs = [docs_and_scores[0][0]]  # fallback to top-1

        # Format retrieved documents for Gemini
        formatted = []
        for doc in filtered_docs:
            meta = doc.metadata
            formatted.append(f"""
    name: {meta.get('name', '')}
    remote_testing: {meta.get('remote_testing', '')}
    adaptive_irt: {meta.get('adaptive_irt', '')}
    assessment_types: {meta.get('assessment_types', '')}
    description: {meta.get('description','')}
    job_levels: {meta.get('job_levels', '')}
    languages: {meta.get('languages', '')}
    assessment_length: {meta.get('assessment_length', '')}
    """)

        joined_docs = "\n".join(formatted)
        prompt = f"""You are an assistant that helps recommend assessments from a list.
    Given the following assessments and the user query, return the names of
    the most relevant assessments — at most 10, at least 1 — in order of relevance.

    Assessments:
    {joined_docs}

    Query: {query}

    Return a list of the top assessments' names only in order.
    Return names only, one per line. No bullets or numbering as I'm parsing this response.
    """
        
        response = model.generate_content(prompt,
        generation_config={
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 1024
        },
        safety_settings=[
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 3},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 3},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 3},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": 3},
        ])
        assessment_names = response.text.strip().splitlines()

        # Clean and lookup final results
        name_to_doc = {doc.metadata['name']: doc for doc in filtered_docs if 'name' in doc.metadata}
        final_docs = [name_to_doc[name.strip()] for name in assessment_names if name.strip() in name_to_doc]

        results = []
        for doc in final_docs:
            meta = doc.metadata
            results.append({
                "name": meta.get("name", ""),
                "url": meta.get("url", ""),
                "remote_testing": meta.get("remote_testing", ""),
                "adaptive_irt": meta.get("adaptive_irt", ""),
                "assessment_length": meta.get("assessment_length", ""),
                "assessment_types": meta.get("assessment_types", ""),
            })

        return jsonify(results)
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({"name":"kkkk","new":"yes"})
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8000)
