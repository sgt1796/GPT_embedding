import time
import backoff
import openai
import logging
from openai import OpenAI, RateLimitError
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import multiprocessing

input_file = "Reviews.csv"
output_file = "embeddings_All.csv"

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
counter = None
nchunks = None

logging.basicConfig(filename='failed_requests.log', level=logging.ERROR)
logging.getLogger('backoff').setLevel(logging.ERROR)


client = OpenAI()

def init(count, chunks):
    global counter, nchunks
    counter, nchunks = count, chunks

@backoff.on_exception(backoff.expo, RateLimitError, max_time=30)
def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

def process_chunk(chunk_index, chunk):
    embeddings = []
    for index, text in enumerate(chunk):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Chunk {chunk_index}, Text Index {index}: {text[:50]}... | Error: {e}")
            embeddings.append(None)
    with counter.get_lock():
        counter.value += 1
    print(f"processing chunk {counter.value}/{nchunks.value} with {len(chunk)} strings", flush=True)
    return embeddings

def retry_failed_requests(lst):
    persistent_failures = []
    with open('failed_requests.log', 'r') as log_file:
        failed_requests = log_file.readlines()

    retry_embeddings = {}
    for line in failed_requests:
        try:
            # handle unexpected values and missing indices
            parts = line.strip().split('|')
            chunk_info = parts[0].strip().split(',')
            chunk_index_str = chunk_info[0].split(' ')[1].strip()
            text_index_str = chunk_info[1].split(' ')[3].strip()

            # Ensure both are valid integers
            if chunk_index_str.isdigit() and text_index_str.isdigit():
                chunk_index = int(chunk_index_str)
                text_index = int(text_index_str)

                # Retry failed request
                chunk_text = lst[chunk_index][text_index]
                embedding = get_embedding(chunk_text)

                if chunk_index not in retry_embeddings:
                    retry_embeddings[chunk_index] = [None] * len(lst[chunk_index])
                retry_embeddings[chunk_index][text_index] = embedding
            else:
                print(f"Skipping line due to parsing issues: {line}")
        except Exception as e:
            print(f"Retry failed for Chunk {chunk_index_str}, Text Index {text_index_str}: {e}")
            persistent_failures.append((chunk_index_str, text_index_str, lst[int(chunk_index_str)][int(text_index)]))

    # Save persistent failures for manual review
    if persistent_failures:
        with open('persistent_failures.log', 'w') as pf_log:
            for failure in persistent_failures:
                pf_log.write(f"Chunk {failure[0]}, Text Index {failure[1]}, Text: {failure[2][:50]}...\n")

    return retry_embeddings

def main():
    global counter, nchunks, lst
    df = pd.read_csv(input_file)
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
        print(f"Pooling start, {pool._processes} processes launched.\n")
        results = pool.starmap(process_chunk, [(i, chunk) for i, chunk in enumerate(lst)])
        embeddings = []
        for result in results:
            embeddings.extend(result)

    df['embedding'] = embeddings
    # Retry failed requests
    print("Retrying failed requests...")
    retry_embeddings = retry_failed_requests(lst)

    # Update the embeddings with the retries
    for chunk_index, chunk_embeddings in retry_embeddings.items():
        for text_index, embedding in enumerate(chunk_embeddings):
            if embedding is not None:
                start_index = chunk_index * chunk_size
                df.at[start_index + text_index, 'embedding'] = embedding
    print("saving to file...")
    df.to_csv(output_file, index=False)

    ## performance benchmark:
    ## 24 processes launched
    ## --- 20000 lines embedded in 370.8594162464142 seconds ---

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"--- {nrow} lines embedded in {(time.time() - start_time)} seconds ---")
