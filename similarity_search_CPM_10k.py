## requires: pytorch, transformer, flash-attn
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse

def main(args):
    model_name = args.EMBEDDING_MODEL
    query = args.query

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, 
                                    trust_remote_code=True, 
                                    torch_dtype=torch.float16).to("cuda")
    df = pd.read_csv(args.embedding_file)
    df['embedding'] = df.embedding.apply(eval).apply(np.array, dtype="f")

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

    def find_kNN(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - cosine(x, y),
        top_n: int = 5
    ) -> tuple[list[str], list[float]]:
        query_embedding = encode([query])[0]
        df['similarity'] = df.embedding.apply(lambda x: relatedness_fn(query_embedding, x))

        idx = df.similarity.nlargest(top_n).index
        similarity = df.loc[idx, ['similarity']]
        content = df.loc[idx, ['combined']]
        
        return content.values, similarity.values


    strings, relatednesses = find_kNN(query, df, top_n=args.top)
    def print_results(strings, relatednesses):
        print('='*(len(query)+13))
        print(f"|| QUERY: {query} ||")
        print('='*(len(query)+13))
        print("\nSearch result:")
        print('-'*50)
        for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
            print(f"Result {i}:")
            print(f"Relatedness: {relatedness}")
            print(f"\nContent: {string}")
            print('-'*50)

    print_results(strings, relatednesses)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, help='Query string')
    parser.add_argument('--top', '-n', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--EMBEDDING_MODEL', '-m', type=str, default="openbmb/MiniCPM-Embedding", help='Hugging Face embedding model (default: "openbmb/MiniCPM-Embedding")')
    parser.add_argument('--embedding_file', '-f', type=str, help='Path to the embedding file')

    args = parser.parse_args()
    main(args)

# Example usage:
# python similarity_search_CPM_10k.py -q '为什么有一年四季？' -f stories/stories_cn_wpwx_ebd.csv