import pandas as pd
from dotenv import load_dotenv
from Embedder import Embedder
from numpy import zeros
import argparse


out_format = "csv"
columns = ["name", "category", "content"]
input_file = "stories/stories_cn_oesz.csv"
output_file = "stories/stories_cn_oesz_ebd_Jina.csv"
minimize = False

def main(args):
    load_dotenv(args.env)
    embedder = Embedder(use_api='jina')
    input_file = args.input_file
    output_file = args.output_file
    columns = args.columns
    out_format = args.out_format

    df = pd.read_csv(input_file)
    print(f"{df.shape[0]} rows loaded from {input_file}")
    print(f"Combining columns: {columns}")
    df.loc[:, 'combined'] = df.apply(lambda x: '|'.join(f"{col}: {str(x[col]).strip()}" for col in columns if pd.notna(x[col])), axis=1)

    print("start embedding...")
    df.loc[:, 'embedding'] = None
    embeddings = []
    for i in range(df.shape[0]):
        if i % 200 == 0:
            print(f"Processing row {i}...")

        try:
            ebd = embedder.get_embedding([df.combined[i]])
        except Exception as e:
            print(f"Exception at row {i}: {e}")
            ebd = zeros(1024, dtype="f")
        embeddings.extend(ebd)
    print(f"embedding done! {len(embeddings)} embeddings generated.")
            
    for i, embedding in enumerate(embeddings):
        df.at[i, 'embedding'] = embedding.tolist()
    # Minimize output columns if requested
    if args.minimize:
        df = df[['combined', 'embedding']]
    df.to_csv(output_file, index=False, mode='w', sep='\t' if out_format == 'tsv' else ',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate embeddings for text data using a local model.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Path to the input file, accepts .csv or .txt file.")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="Path to the output file.")
    parser.add_argument('-c', '--columns', type=str, nargs='+', help="Names of columns to combined and embedded.")
    parser.add_argument('--out_format', type=str, choices=['csv', 'tsv'], default='csv', help="Output format: 'csv' or 'tsv' (default: csv).")
    # parser.add_argument('--chunk_size', type=int, help="Number of rows to load into memory at a time. By default whole file will be load into the memory.")
    parser.add_argument('--minimize', action='store_true', help="Minimize output to only the combined and embedding columns.")
    parser.add_argument('--env', type=str, default='.env', help='Path to the .env file (default: .env)')

    args = parser.parse_args()
    main(args)