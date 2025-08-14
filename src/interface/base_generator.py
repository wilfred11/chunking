from abc import ABC, abstractmethod
from typing import List


class BaseGenerator(ABC):


    @abstractmethod
    def generate_response(self, query: str, context: List[str]) -> str:
        pass

    @abstractmethod
    def generate_summary(self, context: str)->str:
        pass

    @abstractmethod
    def generate_all_q_and_a(self):
        pass

    @abstractmethod
    def generate_summaries(self):
        pass

