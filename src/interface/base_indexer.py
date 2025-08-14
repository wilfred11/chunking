from abc import ABC, abstractmethod
from typing import List

from src.interface.base_datastore import DataItem


class BaseIndexer(ABC):

    @abstractmethod
    def index(self) -> List[DataItem]:
        pass

    @abstractmethod
    def get_max_length_summary(self) -> int:
        pass

    @abstractmethod
    def get_max_length_chunk(self) -> int:
        pass