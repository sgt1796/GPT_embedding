import pandas as pd
from Embedder import Embedder

out_format = "csv"
columns = ["name", "category", "content"]
input_file = "stories/stories_cn_oesz.csv"
output_file = "stories/stories_cn_oesz_ebd_Jina.csv"
minimize = False


embedder = Embedder(use_api='jina')

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

    ebd = embedder.get_embedding([df.combined[i]])
    embeddings.extend(ebd)
print(f"embedding done! {len(embeddings)} embeddings generated.")
        
for i, embedding in enumerate(embeddings):
    df.at[i, 'embedding'] = embedding.tolist()
# Minimize output columns if requested
if minimize:
    df = df[['combined', 'embedding']]
df.to_csv(output_file, index=False, mode='w', sep='\t' if out_format == 'tsv' else ',')