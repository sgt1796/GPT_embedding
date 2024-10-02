from jina import Deployment, Executor, requests
import numpy as np
from docarray import BaseDoc, DocList
from dotenv import load_dotenv
from requests import post as POST
from Embedder import Embedder
from similarity_search_10k import find_kNN, print_results

from docarray.typing.tensor.embedding.embedding import AnyEmbedding

load_dotenv("../.env")
port = 1192

class TestDoc(BaseDoc):
    text: str = None
    embedding: AnyEmbedding #= np.zeros((1000, ))

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
    def jina__search(self, docs, **kwargs):
        embedder = Embedder(use_api = "jina")
        for doc in docs:
            query = doc.text
            df = "stories/stories_cn_oesz_ebd_Jina.csv"
            strings, relatednesses = find_kNN(query, df, embedder, top_n=5, return_content=True)
            doc.strings = strings
            doc.relatednesses = relatednesses

        return docs

dep = Deployment(port=port, name='embedding_executor', uses=RAG_API, host='localhost')

with dep:
    dep.block()