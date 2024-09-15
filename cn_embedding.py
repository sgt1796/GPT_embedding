## requires: pytorch, transformer, flash-attn
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd

out_format = 'csv'
columns = ['title', 'text', 'category']
output_file = 'stories_embeddings.csv'

model_name = "openbmb/MiniCPM-Embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, 
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


df = pd.read_csv("../story_crawler/stories.csv")
df.loc[:, 'combined'] = df.apply(lambda x: '|'.join(f"{col}: {str(x[col]).strip()}" for col in columns if pd.notna(x[col])), axis=1)

df.loc[:, 'embedding'] = None
embeddings = []
for i in range(df.shape[0]):
    if i % 100 == 0:
        print(i)
    embeddings.extend(encode(df.combined[i]))

print("embedding done!")
for i, embedding in enumerate(embeddings):
    df.at[i, 'embedding'] = embedding.tolist()
# print(df.embedding[:40])
df.to_csv(output_file, index=False, mode='w', sep='\t' if out_format == 'tsv' else ',')