import numpy as np
from sklearn.mixture import GaussianMixture

# load embeddings
embeddings = np.load("embeddings.npy")

# choose number of clusters
n_clusters = 20

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type='full',
    random_state=42
)

gmm.fit(embeddings)

# cluster probabilities
cluster_probs = gmm.predict_proba(embeddings)

print("Cluster probability shape:", cluster_probs.shape)
#save the probability of each cluster and retrieve dominant cluster
np.save("cluster_probs.npy", cluster_probs)
dominant_cluster = cluster_probs.argmax(axis=1)
print("Example dominant clusters:", dominant_cluster[:10])
np.save("dominant_clusters.npy", dominant_cluster)