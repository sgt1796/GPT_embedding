import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()

df = pd.read_csv("GBD_2018_20k_embedding.csv")
df["embedding"] = df.embedding.apply(eval).apply(np.array, dtype="f")
# search function
def find_kNN(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["combined"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

#str_to_search = "what pizza flavor do people like the most?"
str_to_search = "Show me the french >70 aged people, with breast cancer"

strings, relatednesses = find_kNN(str_to_search, df, top_n=5)
print('='*(len(str_to_search)+13))
print(f"|| QUERY: {str_to_search} ||")
print('='*(len(str_to_search)+13))
print("\nSearch result:")
print('-'*50)
for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
    print(f"Result {i}:")
    print(f"Relatedness: {relatedness:.3f}")
    print(f"\nContent: {string}")
    print('-'*50)