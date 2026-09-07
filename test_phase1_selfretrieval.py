import pandas as pd
from src.retrieval.faiss_search import FaissSearcher

df = pd.read_parquet("datasets/featured/feature_engineered_dataset.parquet")
sample = df.iloc[0]["func_code_string"]

s = FaissSearcher()
results = s.search(sample, top_k=5)
print(results[["similarity_score"]].head())
print("TOP1_SIMILARITY:", float(results.iloc[0]["similarity_score"]))
