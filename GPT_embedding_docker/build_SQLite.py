import sqlite3
import pandas as pd

db = "Reviews.db"
chunk_size = 10000
embedding_file = "embedding_data/Reviews_embedding.csv"

# initialize db
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''
           CREATE TABLE reviews(
                row_number INTEGER PRIMARY KEY,
                content TEXT
           )
           ''')

# load data to write
chunk_iter = pd.read_csv(embedding_file, usecols=["combined"], chunksize=chunk_size)
if chunk_size == None:
        chunk_iter = [chunk_iter]

chunk_number = 0
for df in chunk_iter:
    print(f"writing file chunk {chunk_number + 1} to database...")
    cur.executemany('INSERT INTO reviews (row_number, content) VALUES (?, ?)', enumerate(df.combined, start=chunk_number*(chunk_size)) )
    chunk_number += 1
        
print("saving...")
con.commit()
con.close()
print("done")