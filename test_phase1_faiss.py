from src.retrieval.faiss_search import FaissSearcher

sample_code = """
def add(a, b):
    return a + b
""".strip()

s = FaissSearcher()
print(s.info())

r = s.search(sample_code, top_k=5)
print(r[["similarity_score"]] if not r.empty else "EMPTY")
