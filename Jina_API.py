from jina import Deployment, Executor, requests
from typing import List, Optional
import numpy as np
import pandas as pd
from docarray import BaseDoc, DocList
from dotenv import load_dotenv
from Embedder import Embedder
from similarity_search_10k import find_kNN
from docarray.typing.tensor.embedding.embedding import AnyEmbedding

load_dotenv("../.env")
port = 1192

class TestDoc(BaseDoc):
    text: str = None
    embedding: Optional[AnyEmbedding] #= np.zeros((1024, ))
    contents: List[str] = []
    relatedness: List[float] = []

class RAG_API(Executor):
    @requests(on='/jina/embedding')
    def jina_embedding(self, docs: DocList[TestDoc], **kwargs) -> DocList[TestDoc]:
        embedder = Embedder(use_api = "jina")
        input_text = [doc.text for doc in docs if doc.text]
        embeddings = embedder.get_embedding(input_text)
        for doc, embedding_data in zip(docs, embeddings):
                doc.embedding = np.array(embedding_data, dtype="f")

        return docs
    
    ## This calculates the cosine similarity between the query and all the embeddings, slow
    @requests(on='/jina/_search')
    def jina__search(self, docs: DocList[TestDoc], **kwargs):
        embedder = Embedder(use_api = "jina")
        for doc in docs:
            query = doc.text
            df = pd.read_csv("stories/stories_cn_oesz_ebd_Jina.csv")
            df['embedding'] = df.embedding.apply(eval).apply(np.array, dtype="f")
            strings, relatednesses = find_kNN(query, df, embedder, top_n=5)
            doc.contents = [string[0] for string in strings if string]
            doc.relatedness = [relatedness for relatedness in relatednesses if relatedness]

        return docs

if __name__ == '__main__':
    dep = Deployment(port=port, name='embedding_executor', uses=RAG_API, host='localhost')

    with dep:
        dep.block()
