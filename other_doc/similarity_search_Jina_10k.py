from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import argparse

from Embedder import Embedder

query = 'Can you give me a bear story?'
top = 5
EMBEDDING_MODEL = "jina-embeddings-v3"
embedding_file = "stories/stories_cn_oesz_ebd_Jina.csv"

df = pd.read_csv(embedding_file)
df['embedding'] = df.embedding.apply(eval).apply(np.array, dtype="f")

def find_kNN(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - cosine(x, y),
    top_n: int = 5
) -> tuple[list[str], list[float]]:
    embedder = Embedder(use_api='jina', model_name=EMBEDDING_MODEL)
    query_embedding = embedder.get_embedding([query])[0]
    df['similarity'] = df.embedding.apply(lambda x: relatedness_fn(query_embedding, x))

    idx = df.similarity.nlargest(top_n).index
    similarity = df.loc[idx, ['similarity']]
    content = df.loc[idx, ['combined']]
    
    return content.values, similarity.values

strings, relatednesses = find_kNN(query, df, top_n=top)
def print_results(strings, relatednesses):
    print('='*(len(query)+13))
    print(f"|| QUERY: {query} ||")
    print('='*(len(query)+13))
    print("\nSearch result:")
    print('-'*50)
    for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
        print(f"Result {i}:")
        print(f"Relatedness: {relatedness}")
        print(f"\nContent: {string}")
        print('-'*50)

print_results(strings, relatednesses)