## requires: pytorch, transformer, flash-attn
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

def main(args):
    out_format = args.out_format
    columns = args.columns
    input_file = args.input_file
    output_file = args.output_file
    EMBEDDING_MODEL = args.EMBEDDING_MODEL
    minimize = args.minimize

    # Rest of the code goes here
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, 
                                    trust_remote_code=True, 
                                    attn_implementation="flash_attention_2", 
                                    torch_dtype=torch.float16).to("cuda")
    model.eval()

    def weighted_mean_pooling(hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    @torch.no_grad()
    def encode(input_texts):
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to("cuda")
        
        outputs = model(**batch_dict)
        attention_mask = batch_dict["attention_mask"]
        hidden = outputs.last_hidden_state

        reps = weighted_mean_pooling(hidden, attention_mask)   
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings


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
        embeddings.extend(encode(df.combined[i]))
    print(f"embedding done! {len(embeddings)} embeddings generated.")
        
    for i, embedding in enumerate(embeddings):
        df.at[i, 'embedding'] = embedding.tolist()
    # Minimize output columns if requested
    if minimize:
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
    parser.add_argument('--EMBEDDING_MODEL', type=str, default='openbmb/MiniCPM-Embedding', help="Local embedding model to use.")

    args = parser.parse_args()
    main(args)