FROM python:3.12-slim



LABEL description="dev enviroment for embedding and search"
ARG device=cpu

# Install necessary system packages and Python dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install pandas python-dotenv openai backoff faiss-${device}

COPY . /embedding
WORKDIR /embedding

EXPOSE 8080

CMD ["bash"]