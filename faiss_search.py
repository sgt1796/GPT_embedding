import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import linecache


load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()

index = faiss.read_index("IVFPQ_index.bin")
# input is a single vector

def faiss_search(
    query: str,
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    D, I = index.search(np.array([query_embedding], dtype='f'), top_n)
    print(I)
    # print(D)
    related_text = []
    for idx, err in zip(I[0], D[0]):
        #print(idx)
        #print(err)
        # print(linecache.getline("embedding_data/Reviews_embedding.csv", idx)[:150])
        if idx <= 5000:
            related_text.append((linecache.getline("embedding_data/embeddings_5k.csv", idx)[:350], err))
    if len(related_text) == 0:
        related_text.append(("No result find in the first 5000 rows.", -1))

    related_text.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*related_text)
    return strings[:top_n], relatednesses[:top_n]

#str_to_search = "what pizza flavor do people like the most?"
str_to_search = "what toys are more favored by boys then girls?"

strings, relatednesses = faiss_search(str_to_search, top_n=5)


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