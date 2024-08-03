import faiss
import numpy as np
import pandas as pd

# Define parameters
dimension = 1536  # Dimension of the vectors
num_centroids = 96  # Number of centroids (clusters)
assert dimension % num_centroids == 0
code_size = 8  # Size of PQ codes
chunk_size = 10000  # Number of vectors to process in each chunk

def load_embeddings_in_chunks(file_path, chunk_size=30000, column="embedding"):
    """
    Load embedding vectors from a CSV file in chunks.
    """
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        embeddings = chunk[column].apply(eval).apply(lambda x: np.array(x, dtype='f'))
        yield np.vstack(embeddings)


def build_ivfpq_index(file_path, chunk_size=30000):
    """
    Build an IVFPQ index from embedding vectors loaded in chunks.
    """
    # Create the quantizer (coarse quantizer)
    quantizer = faiss.IndexFlatL2(dimension)  # The coarse quantizer
    # Create the IVF+PQ index
    index = faiss.IndexIVFPQ(quantizer, dimension, num_centroids, code_size, 8)

    # Train the index with the first chunk
    file_list = load_embeddings_in_chunks(file_path, chunk_size)
    first_chunk = next(file_list)
    print("processing chunk 1...")
    faiss.normalize_L2(first_chunk)
    index.train(first_chunk)
    index.add(first_chunk)
    count = 1
    # Add the rest of the vectors incrementally
    for embeddings in file_list:
        count += 1
        print(f"processing chunk {count}...")
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    return index

file_path = "parallel_embeddings.csv"
index = build_ivfpq_index(file_path, chunk_size=30000)
faiss.write_index(index, "Amazon_fine_food_IVFPQ1.faiss")