import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):
        self.cache = []
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def search(self, query_embedding, query_cluster):

        # Filter cache entries by cluster
        candidates = [
            item for item in self.cache if item["cluster"] == query_cluster
        ]

        if len(candidates) == 0:
            self.miss_count += 1
            return None

        embeddings = np.array([item["embedding"] for item in candidates])

        sims = cosine_similarity([query_embedding], embeddings)[0]

        best_idx = np.argmax(sims)

        if sims[best_idx] >= self.threshold:

            self.hit_count += 1

            result = candidates[best_idx].copy()
            result["similarity"] = float(sims[best_idx])

            return result

        self.miss_count += 1
        return None

    def add(self, query, embedding, result, cluster):

        entry = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

        self.cache.append(entry)

    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }