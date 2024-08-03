import faiss
import numpy as np
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI()
faiss_index = faiss.read_index("Amazon_fine_food_IVFPQ.faiss")

query = "Recommand some dog food for me"
query_embedding = client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding
query_embedding = np.expand_dims(np.array(query_embedding), axis=0).astype('float32')

df = pd.read_csv('embedding_1k.csv')
df["embedding"] = df.embedding.apply(eval).apply(lambda x: np.array(x, dtype='f'))
df_query = np.expand_dims(df.embedding[1], axis=0)
faiss_index.nprobe = 20
D, I = faiss_index.search(df_query, 5)
print(I)
print(D)
