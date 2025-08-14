import os
from typing import List

import duckdb
import pandas as pd
from pyarrow import UuidType

from src.interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load the environment variables from .env file
load_dotenv()


#embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
# https://huggingface.co/thenlper/gte-large
#embedding_model = SentenceTransformer("thenlper/gte-large")


class Datastore(BaseDatastore):
    DB_PATH = "data/rag-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        #self.vector_dimensions = 1536
        self.vector_dimensions = 1024
        self.emb_model: SentenceTransformer = self._get_sentence_transformer()
        #self.open_ai_client = OpenAI( api_key=os.environ.get("OPENAI_API_KEY"))
        #self.open_ai_client = OpenAI(base_url=os.environ.get("LMSTUDIO_URL"), api_key=os.environ.get("LMSTUDIO_KEY"))

        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table: Table = self._get_table()

    def reset(self) -> Table:
        # Drop the table if it exists
        try:
            self.vector_db.drop_table(self.DB_TABLE_NAME)
        except Exception as e:
            print("Unable to drop table. Assuming it doesn't exist.")

        # Create the new table.
        schema = pa.schema(
            [
                pa.field('id', pa.int32()),
                pa.field("summary_vector", pa.list_(pa.float32(), self.vector_dimensions)),
                pa.field("content", pa.utf8()),
                pa.field("source", pa.utf8()),
                pa.field("summary", pa.utf8()),
                pa.field("numbering", pa.utf8()),
                pa.field("question", pa.utf8()),
                pa.field("answer", pa.utf8())
            ]
        )

        self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        self.table = self.vector_db.open_table(self.DB_TABLE_NAME)
        print(f"✅ Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.table

    """def get_vector(self, content: str) -> List[float]:

        response = self.open_ai_client.embeddings.create(
            input=content,
            model="text-embedding-3-small",
            dimensions=self.vector_dimensions,
        )
        embeddings = response.data[0].embedding
        return embeddings"""

    """def get_vector(self, content: str) -> List[float]:

        response = self.open_ai_client.embeddings.create(
            input=content,
            #model="gte-large-gguf/gte-large.Q6_K.gguf",
            model="Nomic-embed-text-v1.5-Embedding-GGUF/nomic-embed-text-v1.5.f16.gguf",
            dimensions=self.vector_dimensions,
        )
        embeddings = response.data[0].embedding
        return embeddings"""

    def get_vector(self, summary: str) -> List[float]:
        #print(content)
        print("length content")
        print(len(summary))
        if not summary.strip():
            print("Attempted to get embedding for empty text.")
            return []

        embedding = self.emb_model.encode(summary)
        #print(embedding.shape)
        #print(type(embedding))
        l = embedding.tolist()
        #print(type(l))
        return l

    def add_items(self, items: List[DataItem]) -> None:
        # Convert items to entries in parallel (since it's network bound).
        with ThreadPoolExecutor(max_workers=8) as executor:
            entries = list(executor.map(self._convert_item_to_entry, items))

        self.table.merge_insert(
            "source"
        ).when_matched_update_all().when_not_matched_insert_all().execute(entries)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        vector = self.get_vector(query)
        results = (
            self.table.search(vector)
            .select(["content", "source"])
            .limit(top_k)
            .to_list()
        )

        result_content = [result.get("content") for result in results]
        return result_content

    def _get_table(self) -> Table:
        try:
            return self.vector_db.open_table(self.DB_TABLE_NAME)
        except Exception as e:
            print(f"Error opening table. Try resetting the datastore: {e}")
            return self.reset()

    def _get_sentence_transformer(self) -> SentenceTransformer:
        try:
            emb_model = SentenceTransformer("./" + os.environ.get("LOCAL_SENTENCE_TRANSFORMER"), trust_remote_code=True)
            print("emb_model loaded local")
            print("Sentence embedding")
            print(emb_model.get_sentence_embedding_dimension())
            print("Context max length")
            print(emb_model.get_max_seq_length())
            return emb_model
        except Exception as e:
            print(f"Error loading sentence transformer. {e}")
            return None

    def _convert_item_to_entry(self, item: DataItem) -> dict:
        """Convert a DataItem to match table schema."""
        summary_vector = self.get_vector(item.summary)
        return {
            "id": item.id,
            "summary_vector": summary_vector,
            "content": item.content,
            "source": item.source,
            "summary": item.summary,
            "numbering": item.numbering,
            "question": item.question,
            "answer": item.answer
        }

    def describe_table(self):
        print(self.vector_db.open_table(self.DB_TABLE_NAME).schema.to_string())

    def get_number_of_records(self):
        #self._get_table().query().where("number= '1.1'").limit(10).to_arrow()
        #rag_table = self._get_table().to_lance()
        self._get_table().to_pandas().head()
        #duckdb.query("SELECT count FROM rag_table")

    def head(self):
        # self._get_table().query().where("number= '1.1'").limit(10).to_arrow()
        # rag_table = self._get_table().to_lance()
        print(self._get_table().to_pandas().head())

    def as_panda(self) -> pd.DataFrame:
        return self._get_table().to_pandas()

    def to_csv(self):
        return self._get_table().to_pandas().to_csv("data/out/datastore/all_data.csv")
