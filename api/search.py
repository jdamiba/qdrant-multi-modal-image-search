import os
import torch
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()

# Get Qdrant connection details from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "house_images"

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

def get_text_embedding(text):
    """Generate embedding for text query using CLIP"""
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    # Normalize the features
    text_embedding = text_features.cpu().numpy()[0]
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    
    return text_embedding

def search_images(query_text, limit=10):
    """Search for images similar to the query text"""
    # Get embedding for the query text
    query_embedding = get_text_embedding(query_text)
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    
    # Format results
    results = []
    for result in search_results:
        # Ensure the image path is correctly formatted
        image_path = result.payload["image_path"]
        # Remove any leading 'images/' if present to avoid duplication
        if image_path.startswith('images/'):
            image_path = image_path[7:]
            
        results.append({
            "image_path": image_path,
            "filename": result.payload["filename"],
            "score": result.score
        })
    
    return results 