import json
import os
import pickle
from typing import List
from src.interface.base_datastore import DataItem
from src.interface.base_indexer import BaseIndexer


class Indexer(BaseIndexer):
    def __init__(self):
        with open('data/out/pkl/final_chunks_sl.pkl', 'rb') as chunks:
            self.final_chunks: dict((str,int),str) = pickle.load(chunks)
        with open('data/out/pkl/chunk_summaries_sl.pkl', 'rb') as chunk_summa:
            self.chunk_summaries: dict((str,int),str) = pickle.load(chunk_summa)
        with open('data/out/pkl/qa_sl.pkl', 'rb') as chunks_qa:
            self.qas: dict((str,int),str) = pickle.load(chunks_qa)

    def index(self) -> List[DataItem]:
        items = []

        for chunk, summary, (file,chunk_number),qa  in zip(self.final_chunks.values(), self.chunk_summaries.values(), self.final_chunks.keys(), self.qas.values()):
            question_answer = json.loads(qa)
            item = DataItem(content=chunk, summary=summary, source=file, number=chunk_number, question=question_answer["question"], answer=question_answer["answer"])
            items.append(item)
        return items


    def get_max_length_summary(self) -> int:
        if not self.chunk_summaries == None:
            return max(self.chunk_summaries.values(), key=len)
        else:
            return -1

    def get_max_length_chunk(self) -> int:
        if not self.final_chunks == None:
            return max(self.final_chunks.values(), key=len)
        else:
            return -1



