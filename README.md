## GPT embedding

### Usage
`GPT_embedding.py` retrieves embedding from GPT embedding API using `text-embedding-3-small` model, which transform input strings into a float vector of 1572 elements.

```
GPT_embedding.py -h
usage: GPT_embedding.py [-h] -i INPUT_FILE -o OUTPUT_FILE [--out_format {csv,tsv}] [-c COLUMNS [COLUMNS ...]] [--chunk_size CHUNK_SIZE] [--minimize] [--process PROCESS]

Generate GPT embeddings for text data.

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Path to the input file, accepts .csv or .txt with tab as separator.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Path to the output file.
  --out_format {csv,tsv}
                        Output format: 'csv' or 'tsv' (default: csv).
  -c COLUMNS [COLUMNS ...], --columns COLUMNS [COLUMNS ...]
                        Column names to combine.
  --chunk_size CHUNK_SIZE
                        Number of rows to load into memory at a time. By default whole file will be load into the memory.
  --minimize            Minimize output to only the combined and embedding columns.
  --process PROCESS     Number of processes to call. Default will be 1 process per vCPU.
```

The embedded data will have added `embedding` column. 

### Search with user query
To search for similarity in small embedding data (<10k) use `similarity_search_5k.py`. In script change the variable to your query
```
# Change this to your string
str_to_search = "what pizza flavor do people like the most?"
```

For larger data where brute force is not possible, `build_FAISS_index.py` can be used to build FAISS index using IVFPQ method. This will significantly reduce the size to query on as well as the query time.
To search with the index, use `faiss_search.py`. In script change the variable to your query
```
# Change this to your string
str_to_search = "what pizza flavor do people like the most?"
```

