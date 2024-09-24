import faiss
from math import sqrt
import numpy as np
import pandas as pd
import argparse

## Setting for Amazon Fine Food Reviews dataset
#chunk_size = 10000
#file_path = "embedding_data/Reviews_embedding.csv"
#df_iter = pd.read_csv(file_path, chunksize=chunk_size)

# nrow of test file: 568428
#nrow = 568428
#dimension = 1536
#nlist = int(sqrt(nrow)) # number of Voronoi cells to divide. lower this increases accuracy, decreases speed
#quantizer = faiss.IndexFlatL2(dimension)
#nsubvec = 96
#assert dimension % nsubvec == 0
#nbits = 8 # data will be clustered to 2^n centroids


## When data is too large for mem to cluster all at once
## needs a way to "fairly" draw samples to represent the whole data
## A way to do this is via reservior sampling

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

def load_df(file_path, chunk_size, columns=['combined', 'embedding']):
    # load/reload iterator
    try:
        df_iter = pd.read_csv(file_path, chunksize=chunk_size, usecols=columns)
    except ValueError:
        raise ValueError("Columns 'combined' and 'embedding' do not exist in the file.")
    if chunk_size is None:
        df_iter = [df_iter]
    return df_iter


def main(args):
    chunk_size = args.chunk_size
    file_path = args.file_path
    nrow = args.nrow
    nlist = args.nlist
    
    
    dimension = args.dimension
    nsubvec = args.nsubvec    
    nbits = args.nbits
    

    # load iterator & check and calculate parameters
    df_iter = load_df(file_path, chunk_size)

    if isinstance(df_iter, list):
        df = df_iter[0]
    else:
        df = next(df_iter) # this progress the iterator by 1

    # Check dimension
    if dimension == -1:
        dimension = len(eval(df.embedding[0]))
    else:
        assert dimension == len(eval(df.embedding[0])), "Dimension of the first embeddings do not match the provided dimension"

    assert dimension % nsubvec == 0, "Dimension must be divisible by nsubvec."

    # Check nrow
    if nrow == -1:
        if chunk_size is None:
            nrow = df.shape[0]
        else:
            raise ValueError("Number of rows in the data file is not provided. Please provide the number of rows in the file when loading data in chunks.")
    else:
        assert nrow == df.shape[0], "Number of rows in the data file does not match the provided number of rows."

    # Check nlist
    if args.nlist is None:
        nlist = int(sqrt(nrow)) # number of Voronoi cells to divide. lower this increases accuracy, decreases speed

    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, nsubvec, nbits)
    assert not index.is_trained

    # reload df iterator & conduct reservior sampling
    df_iter = load_df(file_path, chunk_size)
    sample_list = []
    if args.resvoir_sample != -1:
        print(f"Sampling {args.resvoir_sample} data from file for clustering...")
        sample_list = reservoir_sampling(df_iter, args.resvoir_sample)
    else:
        if chunk_size is None:
            print("No Reservoir Sampling provided. Clustering all data at once.")
            sample_list = df_iter[0]
        else:
            ## If the resvior_sample is not provided, and df is large (chunk_size is not None)
            raise ValueError("Reservoir Sampling is not provided, and the data is too large to cluster all at once. Please provide the number of samples to cluster.")

    # reload iterator & cluster the data
    df_iter = load_df(file_path, chunk_size)

    sample = pd.DataFrame(sample_list, columns=['combined', 'embedding'])
    sample["embedding"] = sample.embedding.apply(eval).tolist()
    index.train(np.array(sample.embedding.tolist(), dtype="f"))
    print("sample clustering complete! Start building index...")

    # reload iterator & generate index
    df_iter = load_df(file_path, chunk_size)
   
    for df in df_iter:
        df["embedding"] = df.embedding.apply(eval).tolist()
        index.add(np.array(df.embedding.tolist(), dtype="f"))
        print(f"{index.ntotal} vector has been processed!")

    print(f"\nTotal {index.ntotal} vectors processed.")
    print("Index building complete! Writing index to file...")
    faiss.write_index(index, args.out_path)
    print("Index saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=None, help="Size of each chunk. If the data is too large to cluster all at once, use this and resvoir_sample to cluster the data in chunks")
    parser.add_argument("--file_path", "-i", type=str, help="Path to the data file")
    parser.add_argument("--out_path", "-o", type=str, help="Path to the output file")
    parser.add_argument("--nrow", type=int, default=-1, help="Number of rows in the data file, needed only if the data is loaded in chunks")
    parser.add_argument("--nlist", type=int, default=None, help="Number of Voronoi cells to divide. lower this increases accuracy, decreases speed. Default is sqrt(nrow)")
    parser.add_argument("--dimension", '-d', type=int, default=-1, help="Dimension of the embeddings, will use the dimension of the first embedding if not provided")
    parser.add_argument("--nsubvec", type=int, help="Number of subvectors divide the embeddings into, dimension must be divisible by nsubvec")
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits for clustering, default is 8")
    parser.add_argument("--resvoir_sample", type=int, default=-1, help="Perform Reservoir Sampling to draw given number of samples to cluster. By default is no sampling. Must use sampling if the chunk_size is provided)")

    args = parser.parse_args()
    main(args)

# Test command
# python build_FAISS_index.py -i stories/stories_cn_ebd.csv -o stories_cn_ebd.index --nsubvec 96