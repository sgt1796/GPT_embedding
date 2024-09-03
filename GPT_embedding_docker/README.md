# GPT embedding docker version 

This docker image contains codes needed for embedding and searching using Openai embedding model. Main scripts are the following:
 - `GPT_embedding.py`: CLI tool that retrieves embedding from GPT embedding API with multiprocessing.
 - `build_FAISS_index.py`: builds FAISS index on the embedding vectors, uses IVFPQ and [reservior sampling](https://gist.github.com/mdouze/92c5bafcf2b91356cf5e799e3889a0e9).
 - `build_SQLite.py`: builds text data into SQLite database.
 - `faiss_search.py`: (currently `faiss_search_CLI.py`) CLI tool that search a user query within the builded embedding data.

The structure of the working directory is the following:
```
.
├── Dockerfile
├── Dockerfile.backup
├── GPT_embedding.py
├── README.md
├── build_FAISS_index.py
├── build_SQLite.py
├── faiss_search_CLI.py
├── similarity_search_5k.py
└── testing_data
    ├── Reviews_1k.csv
    ├── db
    │   └── Reviews.db
    ├── embeddings_1k.csv
    └── index
        └── IVFPQ_index.bin
```



The general workflow looks like this:

```
`raw text (csv, tsv)` 
   ⬇
`GPT_embedding.py` ➡ `text + embeddings (csv, tsv)`
                         ⬇             ⬇
         `build_FAISS_index.py`      `Build_SQLite.py`
                         ⬇             ⬇
    `FAISS index (IVFPQ_index)`      `SQLite database (SQL.db)`
                         ⬇             ⬇
       `User query ➡  `search_faiss_db.py`
                                ⬇
          ==============================================
          || QUERY: I want to eat chocolate ice cream ||
          ==============================================

          Search result:
          --------------------------------------------------
          Result 1:
          Relatedness: 0.826
          
          Content: Summary: Chocolate Lover's Dream|Text: I am a health nut who loves chocolate. This frozen dessert is absolutely fabulous! I can't go a day without it. The flavor is wonderful,           and for 128 calories per pint - it is a chocolate lover's dream come true. I am an Arctic Zero fan for life!
          --------------------------------------------------
          Result 2: ...
```

## Requirement

Openai API key is needed. Save your API key in `.env`, it should look like
```
OPENAI_API_KEY=<your openai key>
```

You can then mount your folder containing `.env` and other data to the docker, or optionally just the `.env` to the docker container
```bash
docker run -it --rm -v $(pwd)/DATA:/embedding/DATA sgt1796/embedding-dev

# or mount only .env
docker run -it --rm -v $(pwd)/.env:/embedding/.env sgt1796/embedding-dev
```
The full testing data set is [Amazon Fine Food Review](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

## Run with testing data

You can test with sample data within the `testing_data` folder. It has a structure like this
```
└── testing_data
    ├── Reviews_1k.csv
    ├── db
    │   └── Reviews.db
    ├── embeddings_1k.csv
    └── index
        └── IVFPQ_index.bin
```
All commands assume you are at working directory (/embedding/), and `.env` is present at /embedding/DATA/.env

You can remove the `--env` argument if you mounted only `.env` to the folder

### Generate embedding from text files (1k sample)

```
# generate embedding
python GPT_embedding.py \
-i testing_data/Reviews_1k.csv \
-o test_embedding.csv --out_format csv \
-c Summary Text \
--chunk_size 500 \
--env DATA/.env
```
(!) This command may hangs indefinitely in a new docker container enviorment. Kill the process with `Ctrl+C` and re-run should solve this problem.

### search embedding with query (1k sample)
The above command should generate test embedding data `test_embedding.csv` in the current directory. You can then use similarity search to search on this small dataset.
(Keep in mind that the dataset is amazon fine food review, relevant questions might yield better results)

```bash
# Similarity search
python similarity_search_5k.py \
-q "What flavor of chocolate do people like the best?" \
--top 5 \
-f test_embedding.csv

# Test correctness using sample embedding file
# This should yield (almost) same result
python similarity_search_5k.py \
-q "What flavor of chocolate do people like the best?" \
--top 5 \
-f testing_data/embeddings_1k.csv
```

### search with database and IVFPQ index
The pre-built faiss IVFPQ index and the sqlite3 database is avaliable at the `testing_data` folder. You can use them to search the **full dataset**.
```
python faiss_search_CLI.py \
-q "What flavor of chocolate do people like the best?" \
--db testing_data/db/Reviews.db \
--index testing_data/index/IVFPQ_index.bin  \
--top 5 \
-v
```

Note this output will be different from the 1k file output. As it search the whole dataset.

