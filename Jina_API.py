from jina import Deployment, Executor, requests
import numpy as np
from docarray import BaseDoc, DocList
from dotenv import load_dotenv
from requests import post as POST
import os

from docarray.typing.tensor.embedding.embedding import AnyEmbedding

load_dotenv("../.env")

class TestDoc(BaseDoc):
    text: str = None
    embedding: AnyEmbedding #= np.zeros((1000, ))

class embedding(Executor):
    @requests(on='/create')
    def JINA(self, docs: DocList[TestDoc], **kwargs) -> DocList[TestDoc]:
        url = 'https://api.jina.ai/v1/embeddings'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("JINAAI_API_KEY")}'
        }

        input_texts = [doc.text for doc in docs if doc.text]
        data = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": input_texts
        }
        response = POST(url, headers=headers, json=data)
        # Process the response
        if response.status_code == 200:
            # Extract embeddings from the response and convert them to NumPy arrays
            embeddings = response.json().get('data', [])
            for doc, embedding_data in zip(docs, embeddings):
                doc.embedding = np.array(embedding_data['embedding'], dtype="f")  # Assign as a NumPy array
        else:
            # Handle any errors appropriately (logging, raising errors, etc.)
            print(f"Error: {response.status_code}, {response.text}")

        return docs


dep = Deployment(port=1129, name='embedding_executor', uses=embedding, host='localhost')

with dep:
    dep.block()