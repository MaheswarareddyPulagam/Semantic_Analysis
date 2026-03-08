🔍🔎 Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system over the 20 Newsgroups dataset. The system combines transformer embeddings, fuzzy clustering, semantic caching, and a FastAPI service to enable efficient query processing while avoiding redundant computations.
The design focuses on semantic understanding of queries rather than keyword matching, allowing the system to recognize similar queries even when phrased differently.
Model Architecture:
Dataset
   ↓
Data Cleaning
   ↓
Sentence Embeddings
   ↓
Vector Database (FAISS)
   ↓
Fuzzy Clustering (Gaussian Mixture Model)
   ↓
Semantic Cache
   ↓
FastAPI Service


This architecture enables efficient semantic search while reducing redundant computation using intelligent caching.

 Model components actions:

Component                  	Reason
Dataset cleaning           	Remove metadata noise
MiniLM embeddings	          Lightweight semantic representation
FAISS	                      Fast vector similarity search
Gaussian Mixture Model	    Soft clustering for overlapping topics
Semantic cache	            Avoid redundant computations
Cluster-aware lookup	      Improve cache scalability

This system demonstrates how semantic embeddings, clustering, caching, and APIs can be combined to build an efficient semantic search system.
The architecture supports:
1)Semantic query understanding.
2)Efficient document retrieval and preview.
3)Reduced computation through intelligent caching.
