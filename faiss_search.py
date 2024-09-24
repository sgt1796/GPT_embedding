import numpy as np
from openai import OpenAI
import faiss
import sqlite3
import argparse
from os.path import exists
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dotenv import load_dotenv

class Embedder:
    def __init__(self, model_name=None, to_cuda=True, client=None, use_openai=True, attn_implementation=None):
        self.use_openai = use_openai
        if use_openai:
            self.model_name = model_name
            if client is None:
                self.client = OpenAI()
            else:
                self.client = client
        else: # Load a PyTorch model and tokenizer

            # The attention implementation to use in the model (if relevant). Can be any of 
            # `"eager"` (manual implementation of the attention), 
            # `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), 
            # or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). 
            # By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.
            self.attn_implementation = attn_implementation
            self.model_name = model_name
            self.to_cuda = to_cuda

            if attn_implementation:
                self.model = AutoModel.from_pretrained(model_name, 
                                    trust_remote_code=True, 
                                    attn_implementation="flash_attention_2", 
                                    torch_dtype=torch.float16).to('cuda' if to_cuda else 'cpu')
            else:
                self.model = AutoModel.from_pretrained(model_name, 
                                    trust_remote_code=True, 

                                    torch_dtype=torch.float16).to('cuda' if to_cuda else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()

    def get_embedding(self, text: str) -> np.ndarray:
        if self.use_openai:
            query_embedding_response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return np.array(query_embedding_response.data[0].embedding, dtype='f')
        else:
            return np.array(self.encode([text])[0], dtype='f')

            
    def weighted_mean_pooling(self, hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    @torch.no_grad()
    def encode(self, input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to('cuda' if self.to_cuda else 'cpu')
        
        outputs = self.model(**batch_dict)
        attention_mask = batch_dict["attention_mask"]
        hidden = outputs.last_hidden_state

        reps = self.weighted_mean_pooling(hidden, attention_mask)   
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
## This version of faiss search requires a sqlite3.Connection object AND an IVFPQ index object
def faiss_search(
    query: str,
    con: sqlite3.Connection,
    index: faiss.Index,
    embedder: Embedder,
    top: int = 5
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    cur = con.cursor()
    query_embedding = embedder.get_embedding(query)

    # I is a list of list of index
    # D is a list of list of error (relatedness)
    D, I = index.search(np.array([query_embedding], dtype='f'), top)
    # print(I)
    # print(D)

    related_text = []

    #### This doesn't work:
    #### row = 1000
    #### cur.execute("SELECT content FROM reviews WHERE row_number=?", (row,)).fetchall()
    #### But this works:
    #### cur.execute("SELECT content FROM reviews WHERE row_number=?", (1000,)).fetchall()

    ### probably b/c how python treat one-element tuple w/ variable differently...

    ### Current workaround is to first
    ### eval(f"({row},)") 

    for row, err in zip(I[0], D[0]):
        ## retrieve corresponding row from db
        input = eval(f"({row},)")
        content = cur.execute('SELECT content FROM reviews WHERE row_number=?', input).fetchone()[0]
        related_text.append((content, err))

    # might not needed
    if len(related_text) == 0:
        related_text.append(("No results found.", -1))

    related_text.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*related_text)
    return strings[:top], relatednesses[:top]

def get_parser():
    parser = argparse.ArgumentParser(
        description='Faiss Search CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    general_group = parser.add_argument_group('General arguments')
    general_group.add_argument('--query', '-q', type=str, required=True, help='Query string')
    general_group.add_argument('--db', type=str, required=True, help='Database file path')
    general_group.add_argument('--index', '-x', type=str, required=True, help='Index file path')
    general_group.add_argument('--top', '-n', type=int, default=5, help='Number of results to return')
    general_group.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')

    # Model selection arguments
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument('--model_name', type=str, default=None, help='Model name (default "text-embedding-3-small" for OpenAI GPT, must specify model for PyTorch)')
    model_group.add_argument('--GPT', action='store_true', help='Use OpenAI GPT model (API key required in --env)')

    # OpenAI GPT specific arguments
    gpt_group = parser.add_argument_group('OpenAI GPT arguments')
    gpt_group.add_argument('--env', type=str, default='.env', help='Path to the .env file with the API key')

    # PyTorch model specific arguments
    pytorch_group = parser.add_argument_group('PyTorch model arguments')
    pytorch_group.add_argument('--cuda', action='store_true', help='Use CUDA instead of CPU')
    pytorch_group.add_argument('--attn_implementation', type=str, default=None,
                               help='Attention implementation for the model ("eager", "sdpa", or "flash_attention_2"). If not specified, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.')

    return parser

def print_result(strings, relatednesses, query, verbose=False):
    
    if verbose:
        if query != None:
            print('=' * (len(query) + 13))
            print(f"|| QUERY: {query} ||")
            print('=' * (len(query) + 13))
        print("\nSearch result:")
        print('-' * 50)
        for i, (string, relatedness) in enumerate(zip(strings, relatednesses), start=1):
            print(f"Result {i}:")
            print(f"Relatedness: {relatedness:.3f}")
            print(f"\nContent: {string}")
            print('-' * 50)
    else:
        if query != None:
            print(f"\nQuery: {query}")
        for i, string in enumerate(strings, start=1):
            print(f"\nResult {i}: {string}")

def main(args):
    if args.GPT:
        ## Load API key from .env file
        if args.env:
            if not exists(args.env):
                print(f"Error: Environment file '{args.env}' not found.")
                raise FileNotFoundError(f"Environment file '{args.env}' not found.")
            load_dotenv(args.env)
        else:
            load_dotenv()

    model_name = args.model_name
    if model_name is None:
        if args.GPT:
            model_name = 'text-embedding-3-small'
        else:
            raise ValueError("Must specify a model name if not using OpenAI GPT.")
    else:
        model_name = args.model_name

    if args.attn_implementation:
        embedder = Embedder(model_name=model_name, 
                            to_cuda=args.cuda, 
                            use_openai=args.GPT, 
                            attn_implementation=args.attn_implementation)
    else:
        embedder = Embedder(model_name=model_name, 
                            to_cuda=args.cuda, 
                            use_openai=args.GPT)
        
    con = sqlite3.connect(args.db)
    index = faiss.read_index(args.index)

    strings, relatednesses = faiss_search(args.query, con, index, embedder, top=args.top)
    print_result(strings, relatednesses, query=args.query, verbose=args.verbose)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)

## Example usage:
## 1. OpenAI GPT: python faiss_search.py -q 'I love apples, any recommendation for apple-related products?' --db Reviews.db --index IVFPQ_index.bin -v --GPT
## 2. MiniCPM: python faiss_search.py -q '为什么熊会冬眠？' --db stories_cn_test.db --index stories_cn_ebd.index -v --model_name openbmb/MiniCPM-Embedding --cuda

## current problem:
## 1. ads get higher relatedness scores (>1?)
## 2. short text seem to get higher relatedness score