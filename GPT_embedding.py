#!/usr/bin/env python

import argparse
import time
import backoff
import logging
import pandas as pd
from openai import OpenAI, RateLimitError
import multiprocessing
from dotenv import load_dotenv
from numpy import sqrt
from sys import exit

load_dotenv()
client = OpenAI()

logging.basicConfig(filename='retry.log', level=logging.ERROR)
logging.getLogger('backoff').setLevel(logging.ERROR)
open('retry.log', 'w').close()


def init(count, chunks, embedding_model):
    global counter, nchunks, EMBEDDING_MODEL
    counter, nchunks = count, chunks
    EMBEDDING_MODEL = embedding_model

@backoff.on_exception(backoff.expo, RateLimitError, max_time=30)
def _get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

def process_chunk(chunk_index, chunk):
    embeddings = []
    for index, text in enumerate(chunk):
        try:
            embedding = _get_embedding(text)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Chunk {chunk_index}, Text Index {index}: {text[:50]}... | Error: {e}")
            embeddings.append(None)
    with counter.get_lock():
        counter.value += 1
    print(f"Processing chunk {counter.value}/{nchunks.value} with {len(chunk)} strings", flush=True)
    return embeddings

def retry_failed_requests(lst):
    persistent_failures = []
    with open('retry.log', 'r') as log_file:
        failed_requests = log_file.readlines()

    retry_embeddings = {}
    for line in failed_requests:
        chunk_text, chunk_index_str, chunk_index_str = None
        try:
            parts = line.strip().split('|')
            chunk_info = parts[0].strip().split(',')
            chunk_index_str = chunk_info[0].split(' ')[1].strip()
            text_index_str = chunk_info[1].split(' ')[3].strip()

            if chunk_index_str.isdigit() and text_index_str.isdigit():
                chunk_index = int(chunk_index_str)
                text_index = int(text_index_str)
                chunk_text = lst[chunk_index][text_index]
                embedding = _get_embedding(chunk_text)

                if chunk_index not in retry_embeddings:
                    retry_embeddings[chunk_index] = [None] * len(lst[chunk_index])
                retry_embeddings[chunk_index][text_index] = embedding
            else:
                print(f"Skipping line due to parsing issues: {line}")
        except Exception as e:
            print(f"Retry failed for Chunk {chunk_index_str}, Text Index {text_index_str}: {e}")
            persistent_failures.append((chunk_index_str, text_index_str, chunk_text))

    if persistent_failures:
        print(f"\nretry failed for {len(persistent_failures)} row(s): ")
        with open('failed.log', 'w') as pf_log:
            for failure in persistent_failures:
                print(f"File Chunk {current_file_chunk}, Chunk {failure[0]}, Text Index {failure[1]}, Text: {failure[2][:50]}...\n")
                pf_log.write(f"File Chunk {current_file_chunk}, Chunk {failure[0]}, Text Index {failure[1]}, Text: {failure[2][:50]}...\n")

    return retry_embeddings

def main(args):
    input_file = args.input_file
    output_file = args.output_file
    out_format = args.out_format
    columns = args.columns
    minimize = args.minimize
    embedding_model = args.EMBEDDING_MODEL
    chunk_size = args.chunk_size
    process = args.process
    

    global counter, nchunks, current_file_chunk
    if chunk_size == None:
        print("--chunk_size is not specified, loading whole file into memory...")
    else:
        print(f"--chunk_size is set to {chunk_size}, loading the first chunk...")
    if process == None:
        process = multiprocessing.cpu_count()
    # Detect file type based on input file suffix
    if input_file.endswith('.csv'):
        chunk_iter = pd.read_csv(input_file, usecols=columns, chunksize=chunk_size)
    elif input_file.endswith('.txt'):
        chunk_iter = pd.read_csv(input_file, sep='\t', usecols=columns, chunksize=chunk_size)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .tsv file.")
    

    if chunk_size == None:
        chunk_iter = [chunk_iter]

    
    first_chunk = True
    current_file_chunk = 1
    print("Start processing...")
    print(f"EMBEDDING_MODEL: {embedding_model}")
    for chunk_df in chunk_iter:
        # Clear retry log file before processing each chunk
        open('retry.log', 'w').close()
        
        print(f"===== processing file chunk {current_file_chunk}... =====\n")
        chunk_df = chunk_df.dropna(subset=columns)
        print("combining text from columns...")
        chunk_df.loc[:, 'combined'] = chunk_df.apply(lambda x: '|'.join(f"{col}: {str(x[col]).strip()}" for col in columns if pd.notna(x[col])), axis=1)

        ## separator df into smaller list of lists of strings, for each worker to handle
        nrow = chunk_df.shape[0]
        sqrt_chunk_n = int(sqrt(nrow))
        chunk_size = int(nrow / sqrt_chunk_n)
        lst = [list(chunk_df.combined[i:min(i+chunk_size,nrow)]) for i in range(0, nrow, chunk_size)]

        if first_chunk: 
            print("sqrt_chunk_n: ", sqrt_chunk_n)
            print("chunk_size: ", chunk_size)

        # Initialize counters
        counter = multiprocessing.Value('i', 0)
        nchunks = multiprocessing.Value('i', len(lst))

        with multiprocessing.Pool(process, initializer=init, initargs=(counter, nchunks, embedding_model)) as pool:
            print(f"\nPooling start, {pool._processes} processes launched.\n", flush=True)
            results = pool.starmap(process_chunk, [(i, chunk) for i, chunk in enumerate(lst)])
            embeddings = []
            for result in results:
                embeddings.extend(result)

        chunk_df.loc[:, 'embedding'] = embeddings

        # Retry failed requests
        print("Retrying failed requests...")
        retry_embeddings = retry_failed_requests(lst)

        # Update the embeddings with the retries
        for chunk_index, chunk_embeddings in retry_embeddings.items():
            for text_index, embedding in enumerate(chunk_embeddings):
                if embedding is not None:
                    start_index = chunk_index * chunk_size
                    chunk_df.at[start_index + text_index, 'embedding'] = embedding

        print("saving to file...")
        # Minimize output columns if requested
        if minimize:
            chunk_df = chunk_df[['combined', 'embedding']]

        # Write to file incrementally
        if first_chunk:
            chunk_df.to_csv(output_file, index=False, mode='w', sep='\t' if out_format == 'tsv' else ',')
            first_chunk = False
        else:
            chunk_df.to_csv(output_file, index=False, mode='a', header=False, sep='\t' if out_format == 'tsv' else ',')

        print("Processing completed! \n")
        current_file_chunk += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GPT embeddings for text data.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Path to the input file, accepts .csv or .txt with tab as separator.")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="Path to the output file.")
    parser.add_argument('--out_format', type=str, choices=['csv', 'tsv'], default='csv', help="Output format: 'csv' or 'tsv' (default: csv).")
    parser.add_argument('-c', '--columns', type=str, nargs='+', help="Column names to combine.")
    parser.add_argument('--chunk_size', type=int, help="Number of rows to load into memory at a time. By default whole file will be load into the memory.")
    parser.add_argument('--minimize', action='store_true', help="Minimize output to only the combined and embedding columns.")
    parser.add_argument('--process', type=int, help="Number of processes to call. Default will be 1 process per vCPU.")
    parser.add_argument('--EMBEDDING_MODEL', type=str, default='text-embedding-3-small', help='OpenAI embedding model (default: text-embedding-3-small)')

    args = parser.parse_args()

    start_time = time.time()
    try:
        main(args)
    except Exception as e:
        logging.error(f"Error while running: {e}")
        exit(1)

    print(f"--- Processing completed in {(time.time() - start_time)} seconds ---")


# test code
# python GPT_embedding.py -i Reviews.csv -o test_1k.txt --out_format tsv -c Summary Text
# run on server
# nohup python3 GPT_embedding/GPT_embedding.py -i Reviews.csv -o /disk3/GPT_embedding_output/Reviews_embedding.csv --out_format csv -c Summary Text --chunk_size 10000 --process 12 > process.log 2>&1 &