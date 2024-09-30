import numpy as np
from openai import OpenAI
import faiss
import sqlite3
import argparse
from os.path import exists
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dotenv import load_dotenv
from Embedder import Embedder
    
## This version of faiss search requires a sqlite3.Connection object AND an IVFPQ index object
def faiss_search(
    query: list[str],
    con: sqlite3.Connection,
    index: faiss.Index,
    embedder: Embedder,
    top: int = 5
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    cur = con.cursor()
    query_embedding = embedder.get_embedding(query)

    # I is a list of list of index
    # D is a list of list of error (relatedness)
    D, I = index.search(np.array(query_embedding, dtype='f'), top)
    # print(I)
    # print(D)

    related_text = []

    #### This doesn't work:
    #### row = 1000
    #### cur.execute("SELECT content FROM reviews WHERE row_number=?", (row,)).fetchall()
    #### But this works:
    #### cur.execute("SELECT content FROM reviews WHERE row_number=?", (1000,)).fetchall()

    ### probably b/c how python treat one-element tuple w/ variable differently...

    ### Current workaround is to first
    ### eval(f"({row},)") 

    for row, err in zip(I[0], D[0]):
        ## retrieve corresponding row from db
        input = eval(f"({row},)")
        result = cur.execute('SELECT content FROM reviews WHERE row_number=?', input).fetchone()
        if result:
            content = result[0]
        else:
            content = "Content not found."
        related_text.append((content, err))

    # might not needed
    if len(related_text) == 0:
        related_text.append(("No results found.", -1))

    related_text.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*related_text)
    return strings[:top], relatednesses[:top]

def get_parser():
    parser = argparse.ArgumentParser(
        description='Faiss Search CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    general_group = parser.add_argument_group('General arguments')
    general_group.add_argument('--query', '-q', type=str, required=True, help='Query string')
    general_group.add_argument('--db', type=str, required=True, help='Database file path')
    general_group.add_argument('--index', '-x', type=str, required=True, help='Index file path')
    general_group.add_argument('--top', '-n', type=int, default=5, help='Number of results to return')
    general_group.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')

    # Model selection arguments
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument('--model_name', type=str, default=None, help='Model name (default "text-embedding-3-small" for OpenAI GPT, must specify model for PyTorch)')
    model_group.add_argument('--use_api', type=str, default=None, help='Use API instead of local model (API key required in --env)')

    # API specific arguments
    gpt_group = parser.add_argument_group('API arguments')
    gpt_group.add_argument('--env', type=str, default='.env', help='Path to the .env file with the API key')

    # PyTorch model specific arguments
    pytorch_group = parser.add_argument_group('PyTorch model arguments')
    pytorch_group.add_argument('--to_cuda', action='store_true', help='Use CUDA instead of CPU')
    pytorch_group.add_argument('--attn_implementation', type=str, default=None,
                               help='Attention implementation for the model ("eager", "sdpa", or "flash_attention_2"). If not specified, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.')

    return parser

def print_result(strings, relatednesses, query, verbose=False):
    
    if verbose:
        if query != None:
            print('=' * (len(query) + 13))
            print(f"|| QUERY: {query} ||")
            print('=' * (len(query) + 13))
        print("\nSearch result:")
        print('-' * 50)
        for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
            print(f"Result {i}:")
            print(f"Relatedness: {relatedness:.3f}")
            print(f"\nContent: {string}")
            print('-' * 50)
    else:
        if query != None:
            print(f"\nQuery: {query}")
        for i, string in enumerate(strings, start=1):
            print(f"\nResult {i}: {string}")

def main(args):
    use_api = args.use_api

    if use_api is not None:
        # Load API key from .env file
        env_file = args.env if args.env else ".env"
        if not exists(env_file):
            print(f"Error: Environment file '{env_file}' not found.")
            raise FileNotFoundError(f"Environment file '{env_file}' not found.")
        load_dotenv(env_file)

    model_name = args.model_name

    if model_name is None:
        if use_api == "":
            model_name = 'text-embedding-3-small'
        elif use_api == "jina":
            model_name = "jina-embeddings-v3"
        elif use_api == "openai":
            model_name = "text-embedding-3-small"
        else:
            raise ValueError("Must specify a model name if not using API.")


    if args.attn_implementation:
        embedder = Embedder(
            model_name=model_name,
            to_cuda=args.cuda,
            use_api=args.use_api,
            attn_implementation=args.attn_implementation
        )
    else:
        embedder = Embedder(
            model_name=model_name,
            to_cuda=args.to_cuda,
            use_api=args.use_api,
        )
        
    con = sqlite3.connect(args.db)
    index = faiss.read_index(args.index)

    strings, relatednesses = faiss_search([args.query], con, index, embedder, top=args.top)
    print_result(strings, relatednesses, query=args.query, verbose=args.verbose)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)

## Example usage:
## 1. OpenAI GPT: python faiss_search.py -q 'I love apples, any recommendation for apple-related products?' --db Reviews.db --index IVFPQ_index.bin -v --use_api openai
## 2. MiniCPM: python faiss_search.py -q '为什么熊会冬眠？' --db stories_cn_test.db --index stories_cn_ebd.index -v --model_name openbmb/MiniCPM-Embedding --to_cuda

## current problem:
## 1. ads get higher relatedness scores (>1?)
## 2. short text seem to get higher relatedness score