import os
import pickle
from typing import List
from src.interface.base_datastore import DataItem
from src.interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk


class Indexer(BaseIndexer):
    def __init__(self):
        with open('data/out/pkl/final_chunks.pkl', 'rb') as chunks:
            self.final_chunks: dict(str,str) = pickle.load(chunks)
        with open('data/out/pkl/chunk_summaries.pkl', 'rb') as chunk_summa:
            self.chunk_summaries: dict(str,str) = pickle.load(chunk_summa)
        with open('data/out/pkl/numbering.pkl', 'rb') as numbering:
            self.numbering: List[str] = pickle.load(numbering)
        with open('data/out/pkl/chunk_length_ba.pkl', 'rb') as chunk_length_before_after:
            self.chunks_length = pickle.load(chunk_length_before_after)

    def index(self, pdf_name: str) -> List[DataItem]:
        items = []

        for chunk, summary, number in zip(self.final_chunks.values(), self.chunk_summaries.values(), self.numbering):
            item = DataItem(content=chunk, summary=summary, source=pdf_name, number=number)
            items.append(item)

        return items

    def get_max_length_summary(self) -> int:
        if not self.chunks_length == None:
            _, y = zip(*self.chunks_length)
            return max(y)
        else:
            return -1

    def get_max_length_chunk(self) -> int:
        if not self.chunks_length == None:
            x, _ = zip(*self.chunks_length)
            return max(x)
        else:
            return -1



