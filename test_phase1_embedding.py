from src.embeddings.codebert_embedding import EmbeddingManager
import numpy as np

sample_code = """
def add(a, b):
    return a + b
""".strip()

m = EmbeddingManager()
v = m.generate_normalized_embedding(sample_code)

print("shape:", v.shape)
print("dtype:", v.dtype)
print("norm:", np.linalg.norm(v))
