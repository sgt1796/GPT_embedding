import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import cosine, euclidean, cityblock
from openai import OpenAI
from dotenv import load_dotenv
from os.path import exists

def main(args):
    ## Load API key from .env file
    if args.env:
        if not exists(args.env):
            print(f"Error: Environment file '{args.env}' not found.")
            raise FileNotFoundError(f"Environment file '{args.env}' not found.")
        load_dotenv(args.env)
    else:
        load_dotenv()

    client = OpenAI()

    df = pd.read_csv(args.embedding_file)
    df["embedding"] = df.embedding.apply(eval).apply(np.array, dtype="f")
    
    def find_kNN(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        query_embedding_response = client.embeddings.create(
            model=args.EMBEDDING_MODEL,
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

    str_to_search = args.query

    strings, relatednesses = find_kNN(str_to_search, df, top_n=args.top)

    def print_results(strings, relatednesses):
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

    print_results(strings, relatednesses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, help='Query string')
    parser.add_argument('--top', '-n', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--EMBEDDING_MODEL', '-m', type=str, default='text-embedding-3-small', help='OpenAI embedding model (default: text-embedding-3-small)')
    parser.add_argument('--embedding_file', '-f', type=str, help='Path to the embedding file')
    parser.add_argument('--env', type=str, default='.env', help='Path to the .env file (default: .env)')

    args = parser.parse_args()
    main(args)
