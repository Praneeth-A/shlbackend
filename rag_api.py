# rag_api.py
import os
import faiss
import pickle
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import google.generativeai as genai
import logging
from flask_cors import CORS
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
       
        # Load FAISS index and docstore
        with open("data/shl_metadata.pkl", "rb") as f:
            
          docstore = pickle.load(f)
        logging.info("check1")
        index = faiss.read_index("data/shl_index.faiss")
        logging.info("check2")

        # LangChain FAISS setup
        vectorstore = FAISS(
        embedding_function=None,
        index=index,
        docstore=InMemoryDocstore(docstore),
        index_to_docstore_id={i: str(i) for i in range(index.ntotal)}
        ,
        )
        logging.info("check3")

        # Gemini setup
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY not found in environment variables.")
            raise Exception("Missing API key.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        onnx_model_path = "onnx_model/"
        sess = ort.InferenceSession("onnx_model/model.onnx", providers=["CPUExecutionProvider"])
    
        logging.info("check4")
        
        onnx_tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
        logging.info("check5")
        
except Exception as e:
    logging.error("❌ Error during app startup", exc_info=True)
    raise e

logging.info("check6")
def embed_query(texts):
    if isinstance(texts, str):
        texts = [texts]
    # Tokenize
    inputs = onnx_tokenizer(texts, return_tensors="np", padding=True,  return_token_type_ids=True ,truncation=True)
    # Prepare inputs for ONNX
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
    # Run inference
    outputs = sess.run(None, ort_inputs)
     # Get the output from last_hidden_state (usually the first output)
    last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, hidden_size)
    # Mean pooling over tokens
    embeddings = np.mean(last_hidden_state, axis=1)  # shape: (batch_size, hidden_size)
    return embeddings

logging.info("check10")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        lQuery=query.lower()
        if "remote" in lQuery:
          query += "Remote Testing: yes"
        elif "adaptive" or "irt" in lQuery:
             query += "Adaptive/IRT: yes"
        logging.info("check11")

        query_embedding = embed_query(query)
        logging.info("check12")

        # Hybrid FAISS approach
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(query_embedding[0], k=20)
        logging.info("check13")

        # Filter based on scores, guarantee at least 1
        threshold = 0.6  # adjust as needed
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
        logging.info("check14")

        response = model.generate_content(prompt      )
        logging.info("check15")

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
@app.route("/")
def home():
    return "Server is up"
