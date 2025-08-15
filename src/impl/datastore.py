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

load_dotenv()

class Datastore(BaseDatastore):
    DB_PATH = "data/rag-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        self.vector_dimensions = 1024
        self.emb_model: SentenceTransformer = self._get_sentence_transformer()
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table: Table = self._get_table()

    def reset(self) -> Table:
        try:
            self.vector_db.drop_table(self.DB_TABLE_NAME)
        except Exception as e:
            print("Unable to drop table. Assuming it doesn't exist.")

        schema = pa.schema(
            [
                pa.field('id', pa.int32()),
                pa.field("summary_vector", pa.list_(pa.float32(), self.vector_dimensions)),
                pa.field("content", pa.utf8()),
                pa.field("source", pa.utf8()),
                pa.field("summary", pa.utf8()),
                pa.field("number", pa.utf8()),
                pa.field("question", pa.utf8()),
                pa.field("answer", pa.utf8())
            ]
        )

        self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        self.table = self.vector_db.open_table(self.DB_TABLE_NAME)
        print(f"Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.table

    def get_vector(self, summary: str) -> List[float]:
        if not summary.strip():
            print("Attempted to get embedding for empty text.")
            return []

        embedding = self.emb_model.encode(summary)
        l = embedding.tolist()
        return l

    def add_items(self, items: List[DataItem]) -> None:
        # Convert items to entries in parallel (since it's network bound).
        with ThreadPoolExecutor(max_workers=8) as executor:
            entries = list(executor.map(self._convert_item_to_entry, items))

        self.table.merge_insert(
            "source"
        ).when_matched_update_all().when_not_matched_insert_all().execute(entries)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.get_vector(query)
        results = (
            self.table.search(vector)
            .select(["summary", "content","source", "question", "number"])
            .limit(top_k)
            .to_list()
        )
        print("distance")
        print(results[0].get("_distance"))
        return results



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
            "number": item.number,
            "question": item.question,
            "answer": item.answer
        }

    def describe_table(self):
        print(self.vector_db.open_table(self.DB_TABLE_NAME).schema.to_string())

    def get_number_of_records(self) -> int:
        arrow_table = self._get_table().to_lance()
        count=duckdb.query("select count(id) FROM arrow_table")
        return count

    def head(self):
        print(self._get_table().to_pandas().head())

    def as_panda(self) -> pd.DataFrame:
        return self._get_table().to_pandas()

    def to_csv(self):
        return self._get_table().to_pandas().to_csv("data/out/datastore/all_data.csv")
