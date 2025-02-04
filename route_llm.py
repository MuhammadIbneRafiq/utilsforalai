import os
from huggingface_hub import snapshot_download

# Download the desired models
# snapshot_download(repo_id="bert-base-uncased")  # Example for BERT
# snapshot_download(repo_id="mistral/Mixtral-8x7B-Instruct-v0.1")  # Example for Mistral

#import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# Initialize Groq models
LLAMA_7B = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-70b-8192"
)

LLAMA_SMALL = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-8b-8192"
)

# Load local models from Hugging Face
bert_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedding_model.encode(text)

def route_query(query: str) -> tuple[str, str, str]:
    """Route query using a simplified matrix factorization approach."""
    # Generate embeddings for the query
    query_embedding = get_embedding(query)
    
    # Calculate similarity scores with local models if needed
    strong_embedding = get_embedding("This is a strong model prompt.")
    weak_embedding = get_embedding("This is a weak model prompt.")

    # Calculate similarity scores
    similarities = {
        "strong": cosine_similarity([query_embedding], [strong_embedding])[0][0],
        "weak": cosine_similarity([query_embedding], [weak_embedding])[0][0]
    }
    
    # Determine which model to use based on similarity
    if similarities["strong"] > similarities["weak"]:
        model_used = "llama3-7b-2048"
        response = LLAMA_7B.invoke([{"role": "user", "content": query}])
        reasoning = "Complex query requiring larger model"
    else:
        model_used = "llama3-3b-2048"
        response = LLAMA_SMALL.invoke([{"role": "user", "content": query}])
        reasoning = "Straightforward query handled by smaller model"
    
    return response, model_used, reasoning

if __name__ == "__main__":
    test_queries = [
        "Explain quantum entanglement in simple terms",
        "What's 2+2?",
        "Write a poem about gravitational waves",
        "Summarize the latest research in fusion energy"
    ]
    
    for query in test_queries:
        response, model, reasoning = route_query(query)
        print(f"Query: {query}")
        print(f"Model: {model} ({reasoning})")
        print(f"Response: {response}\n{'-'*40}")
