import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from os.path import exists
import faiss
import sqlite3
import argparse
from os.path import exists

## This version of faiss search requires a sqlite3.Connection object AND an IVFPQ index object
def faiss_search(
    query: str,
    con: sqlite3.Connection,
    index: faiss.Index,
    client: OpenAI,
    top_n: int = 5,
    EMBEDDING_MODEL: str = "text-embedding-3-small"
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    cur = con.cursor()
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    # I is a list of list of index
    # D is a list of list of error (relatedness)
    D, I = index.search(np.array([query_embedding], dtype='f'), top_n)
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
        content = cur.execute('SELECT content FROM reviews WHERE row_number=?', input).fetchone()[0]
        related_text.append((content, err))

    # might not needed
    if len(related_text) == 0:
        related_text.append(("No results found.", -1))

    related_text.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*related_text)
    return strings[:top_n], relatednesses[:top_n]


def print_result(strings, relatednesses, str_to_search=None, verbose=False):
    
    if verbose:
        if str_to_search != None:
            print('=' * (len(str_to_search) + 13))
            print(f"|| QUERY: {str_to_search} ||")
            print('=' * (len(str_to_search) + 13))
        print("\nSearch result:")
        print('-' * 50)
        for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
            print(f"Result {i}:")
            print(f"Relatedness: {relatedness:.3f}")
            print(f"\nContent: {string}")
            print('-' * 50)
    else:
        if str_to_search != None:
            print(f"\nQuery: {str_to_search}")
        for i, string in enumerate(strings, start=1):
            print(f"\nResult {i}: {string}")


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

    db = args.db
    if not exists(db):
        raise FileNotFoundError("dababase not found")
    con = sqlite3.connect(db)

    index = faiss.read_index(args.index)

    str_to_search = args.query

    strings, relatednesses = faiss_search(str_to_search, con, index, client, top_n=args.top, EMBEDDING_MODEL=args.EMBEDDING_MODEL)
    con.close()
    print_result(strings, relatednesses, str_to_search, verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faiss Search CLI')
    parser.add_argument('--query', '-q', type=str, help='Query string')
    parser.add_argument('--db', type=str, help='Database file path')
    parser.add_argument('--index', '-x', type=str, help='Index file path')
    parser.add_argument('--top', '-n', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--EMBEDDING_MODEL', type=str, default='text-embedding-3-small', help='OpenAI embedding model (default: text-embedding-3-small)')
    parser.add_argument('--env', type=str, default='.env', help='Path to the .env file (default: .env)')

    args = parser.parse_args()

    main(args)
