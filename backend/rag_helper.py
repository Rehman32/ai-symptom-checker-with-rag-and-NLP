#rag_helper.py file 
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pandas as pd

# Load model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline('text-generation', model='gpt2')

# Load medical info CSV (create this file manually with disease info)
disease_info = pd.read_csv('../data/disease_info.csv')  # columns: disease, description

# Convert descriptions to embeddings
corpus = disease_info['description'].tolist()
corpus_embeddings = retriever_model.encode(corpus, convert_to_tensor=False)

# Build FAISS index
index = faiss.IndexFlatL2(len(corpus_embeddings[0]))
index.add(corpus_embeddings)

def get_disease_info(disease_name):
    # Search disease in our custom corpus
    disease_queries = [disease_name + " symptoms treatment causes"]
    query_embedding = retriever_model.encode(disease_queries)
    D, I = index.search(query_embedding, k=1)
    info = corpus[I[0][0]]

    # Use LLM to summarize it
    summary = qa_model(info, max_length=100, num_return_sequences=1)[0]['generated_text']
    return summary
