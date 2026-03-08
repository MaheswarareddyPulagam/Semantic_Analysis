from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from cache import SemanticCache


class QueryRequest(BaseModel):
    query: str


app = FastAPI()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading embeddings...")
embeddings = np.load("embeddings.npy")

print("Loading cluster info...")
dominant_clusters = np.load("dominant_clusters.npy")

print("Loading documents...")
docs = np.load("docs.npy", allow_pickle=True)


# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index ready with:", index.ntotal, "vectors")


# -----------------------------
# Initialize Cache
# -----------------------------
cache = SemanticCache()


# =================================================
# POST /query
# =================================================
@app.post("/query")
def query_search(request: QueryRequest):

    query = request.query

    query_embedding = model.encode([query])[0]

    # Find nearest document to estimate cluster
    distances, indices = index.search(np.array([query_embedding]), 1)
    query_cluster = int(dominant_clusters[indices[0][0]])

    # Check cache with cluster filtering
    cached = cache.search(query_embedding, query_cluster)

    if cached:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["query"],
            "similarity_score": cached["similarity"],
            "result": cached["result"],
            "dominant_cluster": cached["cluster"]
        }

    # FAISS search
    k = 5
    distances, indices = index.search(np.array([query_embedding]), k)

    result_ids = indices[0].tolist()

    results = [
        {
            "doc_id": i,
            "preview": docs[i][:200]
        }
        for i in result_ids
    ]

    cluster = int(dominant_clusters[result_ids[0]])

    cache.add(query, query_embedding, results, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": cluster
    }

# =================================================
# GET /cache/stats
# =================================================
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


# =================================================
# DELETE /cache
# =================================================
@app.delete("/cache")
def clear_cache():

    cache.cache.clear()
    cache.hit_count = 0
    cache.miss_count = 0

    return {"message": "Cache cleared"}