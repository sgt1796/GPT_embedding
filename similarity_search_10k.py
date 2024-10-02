from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import argparse
from Embedder import Embedder

def find_kNN(
    query: str,
    df: pd.DataFrame | str,
    embedder: Embedder,
    relatedness_fn=lambda x, y: 1 - cosine(x, y),
    top_n: int = 5
) -> tuple[list[str], list[float]]:
    
    # if df is path, read the file
    if df is str:
        if not exists(df):
            raise FileNotFoundError(f"File not found: {df}")
        df = pd.read_csv(df)

    query_embedding = embedder.get_embedding([query])[0]
    df['similarity'] = df.embedding.apply(lambda x: relatedness_fn(query_embedding, x))

    idx = df.similarity.nlargest(top_n).index
    similarity = df.loc[idx, ['similarity']]
    content = df.loc[idx, ['combined']]
    
    return content.values, similarity.values

def print_results(strings, relatednesses, query):
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


def main(args):
    df = pd.read_csv(args.embedding_file)
    df['embedding'] = df.embedding.apply(eval).apply(np.array, dtype="f")
    embedder = Embedder(use_api=args.api)

    strings, relatednesses = find_kNN(args.query, df, embedder, top_n=args.top)
    print_results(strings, relatednesses, args.query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, help='Query string')
    parser.add_argument('--top', '-n', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--api', '-m', type=str, help='Select available embedding API: jina (jina-embeddings-v3), openai (text-embedding-3-small)')
    parser.add_argument('--embedding_file', '-f', type=str, help='Path to the embedding file')
    parser.add_argument('--env', type=str, default='.env', help='Path to the .env file (default: .env)')

    args = parser.parse_args()
    main(args)

# Jina:  python similarity_search_10k.py -q 'Can you give me a bear story?' --api jina --embedding_file stories/stories_cn_oesz_ebd_Jina.csv
# OpenAI: python similarity_search_10k.py -q 'what flavor of chocolate do people like?' --api openai --embedding_file embedding_data/embeddings_1k.csv