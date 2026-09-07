from src.retrieval.faiss_search import FaissSearcher

sample_code = """
def add(a, b):
    return a + b
""".strip()

s = FaissSearcher()

for k in [3, 5, 20]:
    r = s.search(sample_code, top_k=k)
    print(f"requested top_k={k} -> got {len(r)} rows")

print()
r20 = s.search(sample_code, top_k=20)
print(r20[["similarity_score", "token_overlap", "structural_similarity", "rerank_score"]])
