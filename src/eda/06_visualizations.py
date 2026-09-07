import pandas as pd
import matplotlib.pyplot as plt

file_path="datasets/raw/train-00000-of-00001.parquet"

df=pd.read_parquet(file_path)

languages = df["language"].value_counts().head(10)

plt.figure(figsize=(10,5))
languages.plot(kind="bar")

plt.title("Top 10 Programming Languages")
plt.xlabel("Language")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("outputs/figures/languages.png")

plt.show()