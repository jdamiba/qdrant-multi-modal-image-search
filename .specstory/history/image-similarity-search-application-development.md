## SpecStory

## Image Similarity Search Application Development (3/12/2025, 12:59:28 PM)

_**User**_

I would like to use this project template and change its functionality. My goal is to create an image simlarity search application for pictures of houses. I want my users to be able to write text prompts like "victorian house" or "brown furniture" and get similar images. So I need to have a way of embedding the pictures of houses that I have in the /images folder into my qdrant cluster, then create a UI that allows users to search. A note about this project's structure- I am mixing Python, in the top level /api folder, with a Next.js project in the app folder. I want to do the actual embedding and upserting in python, and only use typescript for the frontend

---

_**Assistant**_

I'll help you adapt this project template for your image similarity search application. Let's focus on modifying the requirements.txt file first to include the necessary libraries for image processing and embedding.

```requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
requests==2.31.0
python-multipart==0.0.9
fastembed==0.6.0
qdrant-client==1.13.3
python-dotenv==1.0.1
mistralai==1.5.1
torch
transformers
Pillow
clip-by-openai
numpy
tqdm
```

I've added:
1. `clip-by-openai` - OpenAI's CLIP model which is excellent for image-text embeddings
2. `numpy` - For numerical operations
3. `tqdm` - For progress bars when processing multiple images

For your application, you'll need to:

1. Create a Python script in the `/api` folder to:
   - Load images from your `/images` folder
   - Generate embeddings using CLIP
   - Upsert these embeddings into your Qdrant cluster

2. Modify the frontend to:
   - Remove the note upload functionality
   - Focus on text-to-image search
   - Display image results instead of text

Would you like me to help you create the Python script for embedding the images or modify the frontend components next?

---

_**User**_

yes, generate the entire project

---

_**Assistant**_

# Project Structure Overview

I'll create a complete image similarity search application for houses. Let's start by outlining the project structure and then implement each component.

## Project Structure
```
/
├── api/
│   ├── main.py                 # FastAPI server
│   ├── embed_images.py         # Script to embed images and store in Qdrant
│   ├── search.py               # Search functionality
│   └── utils.py                # Utility functions
├── app/                        # Next.js frontend
│   ├── page.tsx                # Main page (modified)
│   ├── components/
│   │   ├── ImageSearch.tsx     # Search component
│   │   └── ImageResults.tsx    # Results display component
│   └── ...                     # Other Next.js files
├── images/                     # Folder for house images
├── .env                        # Environment variables
└── requirements.txt            # Python dependencies
```

Let's implement each component:

## 1. First, let's update the requirements.txt file:

```requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
requests==2.31.0
python-multipart==0.0.9
fastembed==0.6.0
qdrant-client==1.13.3
python-dotenv==1.0.1
torch
transformers
Pillow
ftfy
regex
tqdm
numpy
```

## 2. Now, let's create the Python backend files:

### api/main.py
```python:api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from search import search_images
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="House Image Search API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "House Image Search API"}

@app.get("/search")
async def search(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(10, description="Number of results to return")
):
    try:
        results = search_images(query, limit)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### api/embed_images.py
```python:api/embed_images.py
import os
import torch
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import argparse

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

def get_image_embedding(image_path):
    """Generate embedding for a single image using CLIP"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
        # Normalize the features
        image_embedding = image_features.cpu().numpy()[0]
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        return image_embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def setup_qdrant():
    """Initialize Qdrant client and create collection if it doesn't exist"""
    client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    
    # Check if collection exists, if not create it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=512,  # CLIP embedding size
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
    
    return client

def embed_and_upload_images(image_folder, batch_size=100):
    """Process all images in the folder and upload to Qdrant"""
    client = setup_qdrant()
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_embeddings = []
        batch_payloads = []
        
        for image_path in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}"):
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                # Create relative path for storage
                rel_path = os.path.relpath(image_path, start=os.path.dirname(image_folder))
                
                batch_embeddings.append(embedding)
                batch_payloads.append({
                    "image_path": rel_path,
                    "filename": os.path.basename(image_path)
                })
        
        if batch_embeddings:
            # Upload batch to Qdrant
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(
                    ids=list(range(i, i + len(batch_embeddings))),
                    vectors=batch_embeddings,
                    payloads=batch_payloads
                )
            )
            print(f"Uploaded batch of {len(batch_embeddings)} embeddings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed images and upload to Qdrant")
    parser.add_argument("--image_folder", type=str, default="../images", 
                        help="Path to folder containing images")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    embed_and_upload_images(args.image_folder, args.batch_size)
    print("Image embedding and uploading complete!")
```

### api/search.py
```python:api/search.py
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
        results.append({
            "image_path": result.payload["image_path"],
            "filename": result.payload["filename"],
            "score": result.score
        })
    
    return results
```

### api/utils.py
```python:api/utils.py
import os
from PIL import Image
import io

def get_image_dimensions(image_path):
    """Get the dimensions of an image"""
    with Image.open(image_path) as img:
        return img.size

def is_valid_image(file_path):
    """Check if a file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def create_thumbnail(image_path, max_size=(300, 300)):
    """Create a thumbnail of an image"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None
```

## 3. Now, let's create the frontend components:

### app/page.tsx (Modified)
```typescript:app/page.tsx
import dynamic from "next/dynamic";
import Image from "next/image";

// Import components with dynamic loading to prevent hydration errors and enable code splitting
const ImageSearch = dynamic(() => import("./components/ImageSearch"), {
  ssr: false,
  loading: () => (
    <div className="animate-pulse h-96 w-full bg-gray-100 rounded-lg flex items-center justify-center">
      <div className="text-gray-500">Loading search component...</div>
    </div>
  ),
});

// Add page metadata for better performance
export const metadata = {
  title: 'House Image Search - Find Similar Houses',
  description: "Search for houses by describing features and find visually similar properties.",
};

export default function Home() {
  return (
    <div className="w-full">
      {/* Hero section */}
      <section className="bg-background border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-12">
          <div className="text-center">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground mb-2">
              <a href="https://qdrant.tech" target="_blank" rel="noopener noreferrer" className="text-amaranth hover:text-amaranth/80">Qdrant</a> House Search
            </h1>
            <h2 className="text-lg md:text-xl font-medium text-gray-500 mb-3">
              Visual Property Search Engine
            </h2>
            <div className="max-w-3xl mx-auto mb-4">
              <p className="text-sm text-gray-600 italic">
                Find your dream home by describing what you're looking for — from Victorian architecture to modern interiors, our AI-powered search understands your preferences and finds visually similar properties.
              </p>
            </div>
            <p className="max-w-4xl mx-auto text-md text-gray-600">
              Search our database of house images using natural language powered by <a href="https://qdrant.tech" target="_blank" rel="noopener noreferrer" className="text-amaranth font-medium hover:underline">Qdrant</a> vector search and <a href="https://openai.com/research/clip" target="_blank" rel="noopener noreferrer" className="text-[#2F6FF0] hover:underline">CLIP</a> embeddings.
            </p>
          </div>
        </div>
      </section>
      
      {/* Main app section */}
      <section className="py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="card px-6 pt-6 pb-16 bg-background border border-gray-200 rounded-lg shadow-sm flex flex-col">
            <ImageSearch />
          </div>
        </div>
      </section>
      
      {/* Features section */}
      <section className="py-12 bg-background border-y border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">How It <span className="text-amaranth">Works</span></h2>
            <p className="max-w-2xl mx-auto text-gray-600">
              Our house search engine uses advanced AI technologies to understand both images and text descriptions.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">1</span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">Image Understanding</h3>
                <p className="text-gray-600 flex-grow">
                  Our system analyzes thousands of house images, understanding architectural styles, 
                  interior features, and design elements.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">Powered by CLIP Vision Model</span>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">2</span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">Vector Embeddings</h3>
                <p className="text-gray-600 flex-grow">
                  Images are converted into vector embeddings that capture visual features,
                  while your text queries are converted to the same vector space.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">Using OpenAI's CLIP technology</span>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <div className="card p-6 h-full flex flex-col bg-background border border-gray-200 rounded-lg shadow-sm">
                <span className="absolute -top-4 -left-4 w-12 h-12 bg-amaranth text-background rounded-full flex items-center justify-center text-xl font-semibold shadow-md">3</span>
                <h3 className="text-xl font-semibold text-foreground mb-3 mt-3">Semantic Search</h3>
                <p className="text-gray-600 flex-grow">
                  Find houses by describing what you want in natural language. Our system matches your 
                  description with visually similar properties in our database.
                </p>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <span className="text-sm text-amaranth">Built with Qdrant vector database</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Search examples section */}
      <section className="py-12 md:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Example <span className="text-amaranth">Searches</span></h2>
            <p className="max-w-2xl mx-auto text-gray-600">
              Try these example searches to see how our system works:
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="card p-8 border border-gray-200 rounded-lg shadow-sm bg-background hover:border-l-4 hover:border-l-amaranth transition-all">
              <div className="flex items-center mb-4">
                <svg className="h-8 w-8 text-amaranth mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                <h3 className="text-xl font-semibold text-foreground">Architectural Styles</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Try searching for specific architectural styles like "Victorian house", "Modern farmhouse", 
                "Mid-century modern home", or "Mediterranean villa".
              </p>
              <span className="badge badge-amaranth inline-flex px-2 py-1 text-xs font-medium rounded-md">Style</span>
            </div>
            
            <div className="card p-8 border border-gray-200 rounded-lg shadow-sm bg-background hover:border-l-4 hover:border-l-amaranth transition-all">
              <div className="flex items-center mb-4">
                <svg className="h-8 w-8 text-amaranth mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
                <h3 className="text-xl font-semibold text-foreground">Features & Colors</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Search for specific features like "house with red brick", "home with large windows", 
                "house with swimming pool", or "kitchen with marble countertops".
              </p>
              <span className="badge badge-amaranth inline-flex px-2 py-1 text-xs font-medium rounded-md">Features</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
```

### app/components/ImageSearch.tsx
```typescript:app/components/ImageSearch.tsx
'use client';

import { useState } from 'react';
import Image from 'next/image';

interface SearchResult {
  image_path: string;
  filename: string;
  score: number;
}

export default function ImageSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`/api/search?query=${encodeURIComponent(query)}&limit=20`);
      
      if (!response.ok) {
        throw new Error('Search failed. Please try again.');
      }
      
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error('Search error:', err);
      setError('An error occurred while searching. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Example search suggestions
  const searchSuggestions = [
    "Victorian house with bay windows",
    "Modern minimalist kitchen",
    "Farmhouse with red barn",
    "Coastal beach house",
    "House with swimming pool",
    "Rustic cabin in the woods"
  ];

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    // Submit the form programmatically
    document.getElementById('search-form')?.dispatchEvent(
      new Event('submit', { cancelable: true, bubbles: true })
    );
  };

  return (
    <div className="w-full">
      <h2 className="text-2xl font-bold mb-6">Find Similar Houses</h2>
      
      <form id="search-form" onSubmit={handleSearch} className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-grow">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe the house you're looking for..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amaranth focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            className="bg-amaranth hover:bg-amaranth/90 text-white font-medium py-3 px-6 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        
        {error && (
          <div className="mt-2 text-red-500 text-sm">{error}</div>
        )}
      </form>
      
      {/* Search suggestions */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Try searching for:</h3>
        <div className="flex flex-wrap gap-2">
          {searchSuggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-1 rounded-full transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
      
      {/* Results */}
      {results.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-4">Search Results</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
                <div className="relative h-48 w-full">
                  <Image
                    src={`/images/${result.image_path}`}
                    alt={result.filename}
                    fill
                    style={{ objectFit: 'cover' }}
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  />
                </div>
                <div className="p-3">
                  <p className="text-sm text-gray-500 truncate" title={result.filename}>
                    {result.filename}
                  </p>
                  <p className="text-xs text-gray-400">
                    Score: {(result.score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {loading && (
        <div className="flex justify-center my-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amaranth"></div>
        </div>
      )}
      
      {!loading && results.length === 0 && query && (
        <div className="text-center py-8">
          <p className="text-gray-500">No results found. Try a different search query.</p>
        </div>
      )}
    </div>
  );
}
```

## 4. Let's create a .env file template:

```
# Qdrant Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## 5. Instructions for running the application:

1. First, install the Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a Qdrant instance running. You can use Docker:
   ```
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. Place your house images in the `/images` folder.

4. Run the script to embed images and store them in Qdrant:
   ```
   cd api
   python embed_images.py
   ```

5. Start the FastAPI backend:
   ```
   cd api
   uvicorn main:app --reload
   ```

6. In a separate terminal, start the Next.js frontend:
   ```
   npm run dev
   ```

7. Access the application at http://localhost:3000

This complete application allows users to:
1. Search for house images using natural language descriptions
2. View results with similarity scores
3. Try suggested searches for inspiration

The backend handles:
1. Processing and embedding images using CLIP
2. Storing embeddings in Qdrant
3. Converting text queries to the same vector space for similarity search

You can extend this application by adding features like:
- Filtering by house features (bedrooms, bathrooms, etc.)
- Saving favorite searches
- Uploading custom images to search against
- Adding more detailed property information

---

_**User**_

we need to update our front end user interface to include a text input for our user to enter their query. then we need to embed that query and then use the embedding to retrieve similar images from the qdrant collection and show those to the user

---

_**Assistant**_



Looking at the code we've already created, we actually have most of this functionality implemented! Let's review what we have and make sure it's working as expected:

1. The `ImageSearch.tsx` component already includes:
   - A text input field for users to enter their query
   - A search button that triggers the search
   - Logic to display the results
   - Example search suggestions

2. The search flow works as follows:
   - User enters a text query in the input field
   - Frontend sends the query to the `/api/search` endpoint
   - Backend (in `search.py`) converts the text to an embedding using CLIP
   - Backend searches Qdrant for similar images using the embedding
   - Results are returned to the frontend and displayed in a grid

However, we need to make sure the Next.js API routes are properly set up to forward requests to our FastAPI backend. Let's create that connection:

```typescript:app/api/search/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const query = searchParams.get('query');
  const limit = searchParams.get('limit') || '10';

  if (!query) {
    return NextResponse.json(
      { error: 'Query parameter is required' },
      { status: 400 }
    );
  }

  try {
    // Forward the request to our FastAPI backend
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(
      `${apiUrl}/search?query=${encodeURIComponent(query)}&limit=${limit}`,
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error forwarding search request:', error);
    return NextResponse.json(
      { error: 'Failed to perform search' },
      { status: 500 }
    );
  }
}
```

Let's also update our `.env` file to include the API URL:

```
# Qdrant Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Next.js Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

With these additions, the complete flow will work:

1. User enters a query in the text input
2. The frontend sends the query to the Next.js API route
3. The Next.js API route forwards the request to our FastAPI backend
4. The FastAPI backend:
   - Converts the text query to an embedding using CLIP
   - Searches Qdrant for similar images using the embedding
   - Returns the results to the frontend
5. The frontend displays the results in a grid

The `ImageSearch.tsx` component we created already handles displaying the results with image thumbnails, filenames, and similarity scores.

To summarize the complete workflow:
1. First, run `embed_images.py` to process your house images and store their embeddings in Qdrant
2. Start the FastAPI backend with `uvicorn main:app --reload`
3. Start the Next.js frontend with `npm run dev`
4. Users can now search for houses by entering text descriptions

Is there any specific part of this flow you'd like me to explain in more detail or modify?

---

_**User**_

we get to get /api/py/search

---

_**Assistant**_



I see the issue. We need to adjust our frontend to call the correct API endpoint. Let's update the `ImageSearch.tsx` component to call our Python backend directly at `/api/py/search` instead of the Next.js API route.

```typescript:app/components/ImageSearch.tsx
'use client';

import { useState } from 'react';
import Image from 'next/image';

interface SearchResult {
  image_path: string;
  filename: string;
  score: number;
}

export default function ImageSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Updated to call the Python backend directly
      const response = await fetch(`/api/py/search?query=${encodeURIComponent(query)}&limit=20`);
      
      if (!response.ok) {
        throw new Error('Search failed. Please try again.');
      }
      
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error('Search error:', err);
      setError('An error occurred while searching. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Example search suggestions
  const searchSuggestions = [
    "Victorian house with bay windows",
    "Modern minimalist kitchen",
    "Farmhouse with red barn",
    "Coastal beach house",
    "House with swimming pool",
    "Rustic cabin in the woods"
  ];

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    // Submit the form programmatically
    document.getElementById('search-form')?.dispatchEvent(
      new Event('submit', { cancelable: true, bubbles: true })
    );
  };

  return (
    <div className="w-full">
      <h2 className="text-2xl font-bold mb-6">Find Similar Houses</h2>
      
      <form id="search-form" onSubmit={handleSearch} className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-grow">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe the house you're looking for..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amaranth focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            className="bg-amaranth hover:bg-amaranth/90 text-white font-medium py-3 px-6 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        
        {error && (
          <div className="mt-2 text-red-500 text-sm">{error}</div>
        )}
      </form>
      
      {/* Search suggestions */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Try searching for:</h3>
        <div className="flex flex-wrap gap-2">
          {searchSuggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-1 rounded-full transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
      
      {/* Results */}
      {results.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-4">Search Results</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
                <div className="relative h-48 w-full">
                  <Image
                    src={`/images/${result.image_path}`}
                    alt={result.filename}
                    fill
                    style={{ objectFit: 'cover' }}
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  />
                </div>
                <div className="p-3">
                  <p className="text-sm text-gray-500 truncate" title={result.filename}>
                    {result.filename}
                  </p>
                  <p className="text-xs text-gray-400">
                    Score: {(result.score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {loading && (
        <div className="flex justify-center my-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amaranth"></div>
        </div>
      )}
      
      {!loading && results.length === 0 && query && (
        <div className="text-center py-8">
          <p className="text-gray-500">No results found. Try a different search query.</p>
        </div>
      )}
    </div>
  );
}
```

Now we need to update our `next.config.js` file to proxy requests to our Python backend:

```javascript:next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/py/:path*',
        destination: 'http://localhost:8000/:path*', // Proxy to Python FastAPI backend
      },
    ];
  },
};

module.exports = nextConfig;
```

This configuration will:
1. Proxy all requests from `/api/py/*` to our Python FastAPI backend running at `http://localhost:8000/*`
2. Allow the frontend to call the Python backend directly without CORS issues

With these changes, the frontend will now call the Python backend at `/api/py/search`, which will be proxied to `http://localhost:8000/search`. The Python backend will process the query, search Qdrant, and return the results to be displayed in the frontend.

---

_**User**_

images not displaying in search results - [0]  GET /images/images/Houses Dataset/494_frontal.jpg 404 in 122ms
[0]  ⨯ The requested resource isn't a valid image for /images/images/Houses Dataset/494_frontal.jpg received text/html; charset=utf-8

---

_**Assistant**_



The issue is with the image paths in the search results. Let's fix this by updating both the backend and frontend to handle image paths correctly.

First, let's update the `search.py` file to ensure the image paths are correctly formatted:

```python:api/search.py
# ... existing code ...

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
```

Now, let's update the `ImageSearch.tsx` component to correctly display the images:

```typescript:app/components/ImageSearch.tsx
// ... existing code ...

{results.map((result, index) => (
  <div key={index} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
    <div className="relative h-48 w-full">
      <Image
        src={`/images/${result.image_path}`}
        alt={result.filename}
        fill
        style={{ objectFit: 'cover' }}
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
      />
    </div>
    <div className="p-3">
      <p className="text-sm text-gray-500 truncate" title={result.filename}>
        {result.filename}
      </p>
      <p className="text-xs text-gray-400">
        Score: {(result.score * 100).toFixed(1)}%
      </p>
    </div>
  </div>
))}
```

Additionally, we need to update the `next.config.js` file to properly handle the images:

```javascript:next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    // Add this to allow unoptimized images (helpful for local development)
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/py/:path*',
        destination: 'http://localhost:8000/:path*', // Proxy to Python FastAPI backend
      },
    ];
  },
};

module.exports = nextConfig;
```

Finally, let's also update the `embed_images.py` script to ensure consistent path handling:

```python:api/embed_images.py
# ... existing code ...

for image_path in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}"):
    embedding = get_image_embedding(image_path)
    if embedding is not None:
        # Create relative path for storage
        rel_path = os.path.relpath(image_path, start=os.path.dirname(image_folder))
        # Remove 'images/' prefix if it exists to avoid duplication
        if rel_path.startswith('images/'):
            rel_path = rel_path[7:]
            
        batch_embeddings.append(embedding)
        batch_payloads.append({
            "image_path": rel_path,
            "filename": os.path.basename(image_path)
        })
```

These changes should fix the image display issues by ensuring:
1. The image paths are correctly formatted in the backend
2. The frontend correctly constructs the image URLs
3. Next.js is configured to handle the images properly

---

_**User**_

let's update the readme to be relevant to this project, which is multimodal search over an images dataset.

---

_**Assistant**_



# Qdrant House Image Search

A powerful application for searching house images using natural language descriptions, powered by CLIP embeddings and Qdrant vector search.

## Features

- **Text-to-Image Search**: Describe the house you're looking for in natural language, and find visually similar properties.
- **Multimodal Understanding**: The system understands both architectural styles and specific features like "Victorian house" or "kitchen with marble countertops".
- **Vector Embeddings with CLIP**: Images are converted into vector embeddings that capture visual features, while text queries are converted to the same vector space.
- **Semantic Search with Qdrant**: Find houses that match your description semantically, not just by keywords.

## Architecture

This application is split into two parts:
1. **Frontend**: Next.js application for the user interface
2. **Backend**: FastAPI service for handling embeddings and search

## Technology Stack

- **Frontend**: Next.js with TypeScript and Tailwind CSS
- **Backend**: FastAPI (Python)
- **Embeddings**: OpenAI's CLIP model for multimodal embeddings
- **Vector Database**: Qdrant
- **Image Processing**: PIL (Python Imaging Library)

## Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- Qdrant instance (cloud or self-hosted)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Qdrant Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Next.js Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qdrant-house-search.git
cd qdrant-house-search
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Place your house images in the `/images` folder.

2. Process and embed the images:
```bash
cd api
python embed_images.py
```

3. Start the FastAPI backend:
```bash
cd api
uvicorn main:app --reload
```

4. In a separate terminal, start the Next.js frontend:
```bash
npm run dev
```

5. Open your browser and navigate to `http://localhost:3000`

## Usage Examples

- Search for architectural styles: "Victorian house", "Modern farmhouse", "Mid-century modern home"
- Search for specific features: "House with red brick", "Home with large windows", "Kitchen with marble countertops"
- Search for locations: "Beach house", "Mountain cabin", "Urban apartment"

## API Documentation

When running locally, the API documentation is available at `http://localhost:8000/docs`.

## How It Works

1. **Image Processing**: House images are processed using CLIP to generate vector embeddings that capture visual features.
2. **Vector Storage**: These embeddings are stored in a Qdrant collection.
3. **Query Processing**: When a user enters a text query, it's converted to the same vector space using CLIP.
4. **Similarity Search**: Qdrant finds the most similar image vectors to the query vector.
5. **Result Display**: The matching images are displayed to the user with similarity scores.

## Troubleshooting

### Common Issues

- **Images not displaying**: Ensure image paths are correct and the images are accessible from the frontend
- **Search returns no results**: Check that your Qdrant instance is properly configured and accessible
- **Slow embedding process**: For large image datasets, consider using a GPU for faster processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for the CLIP model
- Qdrant for the vector database
- Next.js team for the frontend framework

---

_**User**_

actually update the file in the repo

---

_**Assistant**_



I'll update the README.md file in the repository to be relevant to this project.

````markdown:README.md
# Qdrant House Image Search

A powerful application for searching house images using natural language descriptions, powered by CLIP embeddings and Qdrant vector search.

## Features

- **Text-to-Image Search**: Describe the house you're looking for in natural language, and find visually similar properties.
- **Multimodal Understanding**: The system understands both architectural styles and specific features like "Victorian house" or "kitchen with marble countertops".
- **Vector Embeddings with CLIP**: Images are converted into vector embeddings that capture visual features, while text queries are converted to the same vector space.
- **Semantic Search with Qdrant**: Find houses that match your description semantically, not just by keywords.

## Architecture

This application is split into two parts:
1. **Frontend**: Next.js application for the user interface
2. **Backend**: FastAPI service for handling embeddings and search

## Technology Stack

- **Frontend**: Next.js with TypeScript and Tailwind CSS
- **Backend**: FastAPI (Python)
- **Embeddings**: OpenAI's CLIP model for multimodal embeddings
- **Vector Database**: Qdrant
- **Image Processing**: PIL (Python Imaging Library)

## Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- Qdrant instance (cloud or self-hosted)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Qdrant Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Next.js Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qdrant-house-search.git
cd qdrant-house-search
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Place your house images in the `/images` folder.

2. Process and embed the images:
```bash
cd api
python embed_images.py
```

3. Start the FastAPI backend:
```bash
cd api
uvicorn main:app --reload
```

4. In a separate terminal, start the Next.js frontend:
```bash
npm run dev
```

5. Open your browser and navigate to `http://localhost:3000`

## Usage Examples

- Search for architectural styles: "Victorian house", "Modern farmhouse", "Mid-century modern home"
- Search for specific features: "House with red brick", "Home with large windows", "Kitchen with marble countertops"
- Search for locations: "Beach house", "Mountain cabin", "Urban apartment"

## API Documentation

When running locally, the API documentation is available at `http://localhost:8000/docs`.

## How It Works

1. **Image Processing**: House images are processed using CLIP to generate vector embeddings that capture visual features.
2. **Vector Storage**: These embeddings are stored in a Qdrant collection.
3. **Query Processing**: When a user enters a text query, it's converted to the same vector space using CLIP.
4. **Similarity Search**: Qdrant finds the most similar image vectors to the query vector.
5. **Result Display**: The matching images are displayed to the user with similarity scores.

## Troubleshooting

### Common Issues

- **Images not displaying**: Ensure image paths are correct and the images are accessible from the frontend
- **Search returns no results**: Check that your Qdrant instance is properly configured and accessible
- **Slow embedding process**: For large image datasets, consider using a GPU for faster processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for the CLIP model
- Qdrant for the vector database
- Next.js team for the frontend framework
````

---

_**User**_

when i deploy this project to vercel, I get a 404 error when i try to to make a request to the /search endpoint

---

_**Assistant**_



The issue is with your Next.js configuration when deploying to Vercel. The current rewrites configuration in your `next.config.js` works locally but needs to be adjusted for Vercel deployment.

Let's update your `next.config.js` file to handle both local development and Vercel deployment:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    API_BASE_URL: process.env.API_BASE_URL || "https://danek-api.fly.dev",
  },
  async rewrites() {
    return [
      {
        source: "/api/py/:path*",
        destination: process.env.NODE_ENV === "development" 
          ? "http://127.0.0.1:8000/:path*"  // Local development
          : process.env.API_BASE_URL + "/:path*",  // Production API URL
      },
    ];
  },
  // Add performance optimizations
  poweredByHeader: false,
  reactStrictMode: true,
  swcMinify: true,
  images: {
    formats: ["image/avif", "image/webp"],
    domains: ["localhost", "your-production-domain.com"],  // Add your production domain
    // Add this to allow unoptimized images (helpful for local development)
    unoptimized: true,
  },
  // Cache API responses
  httpAgentOptions: {
    keepAlive: true,
  },
  experimental: {
    optimizeCss: true,
    scrollRestoration: true,
  },
  compiler: {
    removeConsole:
      process.env.NODE_ENV === "production"
        ? {
            exclude: ["error", "warn"],
          }
        : false,
  },
};

module.exports = nextConfig;
```

For this to work properly on Vercel, you need to:

1. Deploy your FastAPI backend separately (to a service like Fly.io, Heroku, or any other hosting provider)
2. Set the `API_BASE_URL` environment variable in your Vercel project settings to point to your deployed FastAPI backend URL
3. Make sure your FastAPI backend has CORS configured to accept requests from your Vercel domain

In your FastAPI `main.py`, ensure your CORS middleware is configured properly:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app-domain.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Replace `your-vercel-app-domain.vercel.app` with your actual Vercel domain.

---

_**User**_

i don't want to deploy my api separately, i want to take advantage of vercel's python serverless functions

---

_**Assistant**_



I understand you want to use Vercel's serverless functions for your Python API instead of deploying it separately. Let's restructure your project to work with Vercel's serverless functions approach.

Here's how to adapt your project:

## 1. First, let's create a Vercel-compatible API structure:

Create an `api` directory at the root of your project (not inside the `app` directory) for Vercel's serverless functions:

````python:api/search.py
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs
import json
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cpu"  # Vercel functions don't have GPU
model.to(device)

# Initialize Qdrant client
QDRANT_URL = os.environ.get("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
COLLECTION_NAME = "house_images"

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

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Parse query parameters
        query_params = parse_qs(self.path.split('?')[1]) if '?' in self.path else {}
        query_text = query_params.get('query', [''])[0]
        limit = int(query_params.get('limit', ['10'])[0])
        
        if not query_text:
            self.wfile.write(json.dumps({"error": "Query parameter is required"}).encode())
            return
        
        try:
            results = search_images(query_text, limit)
            self.wfile.write(json.dumps({"results": results}).encode())
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        return

def lambda_handler(event, context):
    """AWS Lambda compatible handler"""
    return {
        'statusCode': 200,
        'body': json.dumps(search_images(
            event.get('queryStringParameters', {}).get('query', ''),
            int(event.get('queryStringParameters', {}).get('limit', 10))
        ))
    }
````

## 2. Create a `vercel.json` file at the root of your project:

````json:vercel.json
{
  "version": 2,
  "builds": [
    { "src": "api/*.py", "use": "@vercel/python" },
    { "src": "package.json", "use": "@vercel/next" }
  ],
  "routes": [
    { "src": "/api/py/(.*)", "dest": "/api/$1" }
  ]
}
````

## 3. Update your `ImageSearch.tsx` component to use the correct endpoint:

````typescript:app/components/ImageSearch.tsx
// ... existing code ...

const handleSearch = async (e: React.FormEvent) => {
  e.preventDefault();
  
  if (!query.trim()) {
    setError('Please enter a search query');
    return;
  }
  
  setLoading(true);
  setError('');
  
  try {
    // This will work both locally and on Vercel
    const response = await fetch(`/api/py/search?query=${encodeURIComponent(query)}&limit=20`);
    
    if (!response.ok) {
      throw new Error('Search failed. Please try again.');
    }
    
    const data = await response.json();
    setResults(data.results);
  } catch (err) {
    console.error('Search error:', err);
    setError('An error occurred while searching. Please try again.');
  } finally {
    setLoading(false);
  }
};

// ... rest of the component ...
````

## 4. Create a `requirements.txt` file in the `api` directory:

````text:api/requirements.txt
torch==2.0.1
transformers==4.30.2
qdrant-client==1.3

---



---

