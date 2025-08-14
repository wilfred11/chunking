## Chunking, lancedb, vector database

https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037
https://medium.com/axinc-ai/sentencetransformer-text-embeddings-model-4a7bac6c2cbf
https://www.marqo.ai/course/introduction-to-sentence-transformers
https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

### Chunking

In this project I am using langchain to convert a relatively simple pdf into text chunks.
I am connecting to my local llm service and ask gemma to summarize each chunk. 

### LanceDB

LanceDB is an embedded database that persists vectorial and non-vectorial data. In this case I will use it to convert chunk summaries to a vector, to store in a lancedb.

The vector can be used to perform a semantic search. By applying some simple vector arithmetic to the embeddings, lanceDB allows us to manipulate the embeddings in ways that capture complex relationships and nuances in the data.


