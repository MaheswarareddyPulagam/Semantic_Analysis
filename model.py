from sentence_transformers import SentenceTransformer
from clean import load_dataset
import numpy as np
docs = np.load("docs.npy", allow_pickle=True)
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings using batching
embeddings = model.encode(
    docs,
    batch_size=64,
    show_progress_bar=True
)

print("Total embeddings:", len(embeddings))
print("Embedding length:", len(embeddings[0]))
np.save("embeddings.npy", embeddings)