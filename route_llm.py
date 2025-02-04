import os
from langchain_groq import ChatGroq
from routellm.controller import Controller
from typing import Tuple

# Configure Groq API keys
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# Initialize Groq models
LLAMA_7B = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-7b-2048"
)

LLAMA_SMALL = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-3b-2048"  # Assuming a smaller model is available
)

# Custom wrapper to use Groq models with RouteLLM's logic
class GroqWrapper:
    def __init__(self, model):
        self.model = model

    def generate(self, query):
        return self.model(query)

# Initialize RouteLLM controller with custom wrappers
client = Controller(
    routers=["mf"],  # Matrix factorization router
    strong_model=GroqWrapper(LLAMA_7B),
    weak_model=GroqWrapper(LLAMA_SMALL),
)

def route_query(query: str) -> Tuple[str, str, str]:
    """Route query using RouteLLM's decision engine"""
    # Use RouteLLM to decide which model to use
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": query}]
    )
    
    # Determine which model was used
    if response.model == "strong":
        model_used = "llama3-7b-2048"
    else:
        model_used = "llama3-3b-2048"
    
    # Extract response and reasoning
    response_text = response.choices[0].message.content
    reasoning = ("Complex query requiring larger model" 
                 if model_used == "llama3-7b-2048" 
                 else "Straightforward query handled by smaller model")
    
    return response_text, model_used, reasoning

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
