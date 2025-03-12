from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from search import search_images
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="Multi-Modal Home Search API")

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