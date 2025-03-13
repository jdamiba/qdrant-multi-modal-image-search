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
