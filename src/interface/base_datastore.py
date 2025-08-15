import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class DataItem(BaseModel):
    id: int=1
    content: str = ""
    source: str = ""
    summary: str = ""
    number: int = -1
    summary_vector: str = ""
    question: str = ""
    answer: str = ""


class BaseDatastore(ABC):
    @abstractmethod
    def add_items(self, items: List[DataItem]) -> None:
        pass

    @abstractmethod
    def get_vector(self, content: str) -> List[float]:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def describe_table(self):
        pass

    @abstractmethod
    def get_number_of_records(self)->int:
        pass

    @abstractmethod
    def head(self):
        pass

    @abstractmethod
    def as_panda(self) -> pd.DataFrame:
        pass
