**Updates 9.30:**

1. **Embedder Class Development**:
   - Created an `Embedder` class to support both API-based and local embedding methods.
   - Refactor most of the code to use Embedder class

2. **Models Tested**:
   - **text-embedding-3-small**: 
     - Provided by OpenAI, accessed via API.
     - Performance: Very fast (300RPM) with good accuracy in English.
   - **jina-embeddings-v3**: 
     - Provided by JinaAI, accessed via API.
     - Performance: Slower (60 RPM) but supports multiple languages.
   - **MiniCPM-embedding**:
     - Runs locally using Torch.
     - Performance: Slow.
   - **Llama3-chinese**:
     - Runs locally using Torch.
     - Performance: Very slow, specifically for Chinese embeddings.

3. **Faiss search generalization**:
   - Faiss search now support both API query embedding and local model query embedding

5. **Jina Embedding Script**:
   - Jina embeddings is current running on stories collected (1 stories per sec due to rate limit)

6. **Client-Server Setup with Jina**:
   - Tested building a client-server API architecture using Jina.
   - Might be helpful in production?
