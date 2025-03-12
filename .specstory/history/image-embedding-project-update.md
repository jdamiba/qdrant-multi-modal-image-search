## SpecStory

## Image Embedding Project Update (3/12/2025, 12:59:28 PM)

_**User**_

I want to update this project to create embeddings of images

---

_**Assistant**_

I'll help you modify the embeddings.py file to handle image embeddings. For this, we'll use the CLIP model from OpenAI (via transformers library) which can generate embeddings for both images and text. Here's how we can update the file:

```python:api/embeddings.py
from fastembed import TextEmbedding
from fastapi import HTTPException
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

# Initialize the embedding models
text_embedder = None
image_embedder = None
image_processor = None

def get_text_embedder():
    """
    Get or initialize the text embedding model
    """
    global text_embedder
    if text_embedder is None:
        try:
            text_embedder = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize text embedding model: {str(e)}")
    return text_embedder

def get_image_embedder():
    """
    Get or initialize the image embedding model and processor
    """
    global image_embedder, image_processor
    if image_embedder is None:
        try:
            model_name = "openai/clip-vit-base-patch32"
            image_embedder = CLIPModel.from_pretrained(model_name)
            image_processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize image embedding model: {str(e)}")
    return image_embedder, image_processor

async def generate_text_embedding(text):
    """
    Generate embedding for a text
    """
    try:
        embedder = get_text_embedder()
        embedding = list(embedder.embed([text]))[0]
        return embedding.tolist()  # Convert numpy array to list for JSON serialization
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding generation error: {str(e)}")

async def generate_image_embedding(image_bytes):
    """
    Generate embedding for an image
    Args:
        image_bytes (bytes): Raw image bytes
    Returns:
        list: Image embedding as a list
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get models
        model, processor = get_image_embedder()
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
        # Convert to list and normalize
        embedding = image_features[0].numpy().tolist()
        return embedding
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image embedding generation error: {str(e)}")
```

This update adds the following capabilities:

1. Added CLIP model initialization for image embeddings
2. Renamed the original embedding function to `generate_text_embedding`
3. Added new `generate_image_embedding` function that accepts image bytes and returns embeddings
4. Added necessary imports for image processing (PIL, io, torch)

You'll need to add these dependencies to your requirements.txt file:

```text:requirements.txt
// ... existing code ...
torch
transformers
Pillow
// ... existing code ...
```

The CLIP model is particularly useful because:
1. It's trained to understand both images and text in the same embedding space
2. It's robust and widely used
3. It can handle various image formats

To use the image embedding function, you would pass it the raw bytes of an image file, and it will return an embedding vector that you can use for similarity comparisons, searching, etc.

Note that the first time you run this, it will download the CLIP model which might take a few moments depending on your internet connection.

---



---

