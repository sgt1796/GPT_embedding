from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import multiprocessing
import time

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
counter = None
nchunks = None
global_request_counter = multiprocessing.Value('i', 0)
request_lock = multiprocessing.Lock()


client = OpenAI()

def init(count, chunks):
    global counter, nchunks
    counter, nchunks = count, chunks
    
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def process_chunk(chunk, RPM = 3000):
    embeddings = []
    for text in chunk:
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.03)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... | Error: {e}")
            embeddings.append(None)
    with counter.get_lock():
        counter.value += 1
    print(f"processing chunk with {len(chunk)} strings, {counter.value}/{nchunks.value} processed.", flush=True)
    return embeddings

def main():
    global counter, nchunks
    df = pd.read_csv("Reviews_20k.csv")
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    global nrow 
    nrow = df.shape[0]
    sqrt_chunk_n = int(np.sqrt(nrow))
    chunk_size = int(nrow / sqrt_chunk_n)
    lst = [list(df.combined[i:min(i+chunk_size,nrow)]) for i in range(0, nrow, chunk_size)]
    print("sqrt_chunk_n: ", sqrt_chunk_n)
    print("chunk_size: ", chunk_size)

    print("start processing...")
    counter = multiprocessing.Value('i', 0)
    nchunks = multiprocessing.Value('i', len(lst))
    with multiprocessing.Pool(initializer=init, initargs=(counter, nchunks)) as pool:
        results = pool.map(process_chunk, lst)
        embeddings = []
        for result in results:
            embeddings.extend(result)

    df['embedding'] = embeddings
    df.to_csv('embeddings_20k.csv', index=False)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"--- {nrow} lines embedded in {(time.time() - start_time)} seconds ---")
