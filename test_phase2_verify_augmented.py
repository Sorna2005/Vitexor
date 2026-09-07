import pandas as pd
from src.retrieval.faiss_search import FaissSearcher

s = FaissSearcher()

print("metadata_path used:", s.metadata_path)
print("func_code_string in columns:", "func_code_string" in s.metadata.columns)
print(s.info())

df = pd.read_parquet("datasets/featured/feature_engineered_dataset.parquet")
sample = df.iloc[0]["func_code_string"]

results = s.search(sample, top_k=5)
print()
print(results[["similarity_score"]].head())
print("TOP1_SIMILARITY:", float(results.iloc[0]["similarity_score"]))

top1_func_code = results.iloc[0].get("func_code_string", None)
print("TOP1_HAS_FUNC_CODE_STRING:", top1_func_code is not None and len(str(top1_func_code)) > 0)
