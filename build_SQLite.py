import sqlite3
import pandas as pd
import argparse

def main(args):
    db = args.db
    chunk_size = args.chunk_size
    embedding_file = args.embedding_file

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
            chunk_size = 1

    chunk_number = 0
    for df in chunk_iter:
        print(f"writing file chunk {chunk_number + 1} to database...")
        cur.executemany('INSERT INTO reviews (row_number, content) VALUES (?, ?)', enumerate(df.combined, start=chunk_number*(chunk_size)) )
        chunk_number += 1
            
    print("saving...")
    con.commit()
    con.close()
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--db", type=str, help="Path of the output database file")
    parser.add_argument("--chunk_size", type=int, default=None, help="Use this if the data is too large to write all at once")
    parser.add_argument("-f", "--embedding_file", type=str, help="Path to the input embedding file")    
    args = parser.parse_args()

    main(args)