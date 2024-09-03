FROM python:3.12
LABEL description="dev enviroment for embedding"
ARG device=gpu
RUN pip install pandas python-dotenv openai backoff faiss-${device}
WORKDIR /embedding


CMD ["bash"]
