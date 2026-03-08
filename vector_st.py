import faiss
import numpy as np

# load embeddings
embeddings = np.load("embeddings.npy")

dimension = embeddings.shape[1]

# create FAISS index
index = faiss.IndexFlatL2(dimension)
#save FAISS index to load API calls faster
# add vectors to index
index.add(embeddings)
faiss.write_index(index, "vector_index.faiss")
print("Total vectors in index:", index.ntotal)
