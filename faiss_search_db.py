import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from os.path import exists
import faiss
import sqlite3

db = "Reviews.db"
if not exists(db):
        exit(1)

con = sqlite3.connect(db)



load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()

index = faiss.read_index("IVFPQ_index.bin")
# input is a single vector


## This version of faiss search requires a sqlite3.Connection object
def faiss_search(
    query: str,
    con: sqlite3.Connection,
    top_n: int = 5
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


str_to_search = "I want to prepare for the thanks-giving dinner. What dishes should I make?"
strings, relatednesses = faiss_search(str_to_search, con, top_n=5)
con.close()

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