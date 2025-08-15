import os
from dotenv import load_dotenv
from src.interface.base_datastore import BaseDatastore
from src.interface.base_retriever import BaseRetriever
import cohere

load_dotenv()

cohere_key= os.environ.get("LMSTUDIO_KEY")

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 3) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k)
        result_content = [d["content"] for d in search_results]
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(
        self, query: str, search_results: list[dict], top_k: int = 10
    ) -> list[dict]:
        result_content = [d["content"] for d in search_results]

        co = cohere.ClientV2(api_key=cohere_key)
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=result_content,
            top_n=top_k,
        )

        result_indices = [result.index for result in response.results]
        print(f"✅ Reranked Indices: {result_indices}")
        return [search_results[i] for i in result_indices]
