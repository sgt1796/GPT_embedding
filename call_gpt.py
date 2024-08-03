from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI()
df = pd.read_csv("Reviews_1k.csv")
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# define a function that process a list of strs
def process_chunk(chunk):
    embeddings = []
    for text in chunk:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

# Chunking the documents
nrow = df.shape[0]
sqrt_chunk_n = int(np.sqrt(nrow))
chunk_size = int(nrow/sqrt_chunk_n)
[list(df.combined[i:i+chunk_size]) for i in range(0,nrow,chunk_size)]
lst = [list(df.combined[i:min(i+chunk_size,nrow)]) for i in range(0,nrow,chunk_size)]
print("sqrt_chunk_n: ", sqrt_chunk_n)
print("chunk_size: ", chunk_size)

## processing 250MB (500k lines) in ~3.5h at a cost of $1.5
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
   futures = [executor.submit(process_chunk, chunk) for chunk in lst]
   embeddings = []
   for future in concurrent.futures.as_completed(futures):
      embeddings.extend(future.result())
df['embedding'] = embeddings
#df.to_csv('parallel_embeddings.csv', index=False)
df.to_csv('embedding_1k.csv', index=False)