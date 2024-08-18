import faiss
from math import sqrt
import numpy as np
import pandas as pd

chunk_size = 10000
file_path = "embedding_data/Reviews_embedding.csv"
df_iter = pd.read_csv(file_path, chunksize=chunk_size)

# nrow of test file: 568428
nrow = 568428
dimension = 1536
nlist = int(sqrt(nrow)) # number of Voronoi cells to divide. lower this increases accuracy, decreases speed
quantizer = faiss.IndexFlatL2(dimension)
nsubvec = 96
assert dimension % nsubvec == 0
nbits = 8 # data will be clustered to 2^n centroids

# This code is taken from: https://gist.github.com/mdouze/92c5bafcf2b91356cf5e799e3889a0e9
def reservoir_sampling(src, nsample, temp_fac=1.5, rs=None): 
    """
    samples nsample vectors from an iterator src that yields matrices
    nsample * temp_fac is the max size of the temporary buffer.
    rs is a RandomState object   
    """
    if rs is None: 
        rs = np.random
    maxsize = int(nsample * temp_fac)
    
    reservoir = []      # represented as a list of subsampled matrices
    nreservoir = 0      # size of the reservoir
    nseen = 0           # number of vectors seen so far 
    threshold = 1.0     # probability for a vector to be included in the reservoir
    
    for mat in src:
        n = len(mat)
        
        if nseen + n < maxsize: 
            # so far, no need to sub-sample
            reservoir.append(mat)
            nreservoir += n
        else: 
            # sample from the input matrix
            mask = rs.rand(n) < threshold
            mat_sampled = mat[mask]
            # add to reservoir
            nreservoir += len(mat_sampled)
            reservoir.append(mat_sampled)
            
            if nreservoir > maxsize: 
                # resamlpe reservoir to nsample
                reservoir = np.vstack(reservoir)
                idx = rs.choice(nreservoir, size=nsample, replace=False)
                reservoir = [reservoir[idx]]
                nreservoir = nsample
                # update threshold
                threshold = nsample / (nseen + n)
            
        nseen += n
    
    # do a last sample
    reservoir = np.vstack(reservoir)
    if nreservoir > nsample: 
        idx = rs.choice(nreservoir, size=nsample, replace=False)
        reservoir = reservoir[idx]
    return reservoir    



index = faiss.IndexIVFPQ(quantizer, dimension, nlist, nsubvec, nbits)
assert not index.is_trained

print("Sampling data from large file for clustering...")
sample_list = reservoir_sampling(df_iter, 20000)
sample = pd.DataFrame(sample_list, columns=['Summary','Text','combined','embedding'])
sample["embedding"] = sample.embedding.apply(eval).tolist()
index.train(np.array(sample.embedding.tolist(), dtype="f"))
print("sample clustering complete!")

# reload iterator
df_iter = pd.read_csv(file_path, chunksize=chunk_size)

first = True
for df in df_iter:
    df["embedding"] = df.embedding.apply(eval).tolist()
    if first:
        test_data = np.array([df.embedding[1997]], dtype='f')
        first = False

    # index.train(np.array(df.embedding.tolist(), dtype="f"))
    index.add(np.array(df.embedding.tolist(), dtype="f"))
    print(f"{index.ntotal} vector has been processed!")

print(f"\nTotal {index.ntotal} processed.")
# input is a single vector
k = 2
D, I = index.search(test_data, k)

print(f"searching for vec: {test_data}")
print(f"result: {I[0]}")
print(f"error: {D[0]}")
faiss.write_index(index, "IVFPQ_index.bin")