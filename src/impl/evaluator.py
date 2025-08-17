import csv
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import pandas as pd

from src.interface.base_evaluator import BaseEvaluator, EvaluationResult
from src.interface.base_generator import BaseGenerator
from src.interface.base_retriever import BaseRetriever
from src.util import invoke_ai_json, AnswerEvaluation

#from src.util.invoke_ai import invoke_ai
#from src.util.extract_xml import extract_xml_tag

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
The question will be provided in <question>...</question> tags.
The response will be provided in <response>...</response> tags.
The expected answer will be provided in <expected_answer>...</expected_answer> tags.

The response doesn't have to exactly match all the words/context the expected answer. It just needs to be right about
the answer to the actual question itself.

Evaluate whether the response is correct or not, and add your reasoning.
Return your evaluation in a json object containing 2 fields, reasoning and is_correct, is_correct should contain 'true' or 'false' and reasoning a short text.
"""


class Evaluator(BaseEvaluator):
    def __init__(self, retriever: BaseRetriever, q_and_as, generator: BaseGenerator):
        self.retriever = retriever
        self.q_and_as = q_and_as
        self.generator = generator

    def evaluate(self):
        questions = [item["question"] for item in self.q_and_as]
        expected_answers = [item["answer"] for item in self.q_and_as]
        files = [item["file"] for item in self.q_and_as]
        chunk_numbers = [item["chunk_number"] for item in self.q_and_as]
        q_numbers = range(1, len(files) + 1)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results: List[EvaluationResult] = list(
                executor.map(
                    self._evaluate_single_question,
                    questions,
                    expected_answers,
                    files,
                    chunk_numbers,
                    q_numbers
                )
            )
        fields = ['question', 'response', 'expected answer', 'source', 'chunk_number', 'is_correct', 'reasoning']
        with open('data/out/eval/qa.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)  # Write header
            for result in results:
                writer.writerow([result.question, result.response, result.expected_answer, result.source, result.chunk_nummber, result.is_correct, result.reasoning])

        number_correct = sum(result.is_correct for result in results)
        print(f"✨ Total Score: {number_correct}/{len(results)}")

    def _evaluate_single_question(
            self, question: str, expected_answer: str, source: str, chunk_number: int, q_number: int
    ) -> EvaluationResult:
        # Evaluate a single question/answer pair.
        response = self.process_query(question, source, chunk_number, q_number)

        result = self._evaluate(question, response, expected_answer)

        result_dict = json.loads(result)

        return EvaluationResult(
            question=question,
            response=response,
            expected_answer=expected_answer,
            source=source,
            chunk_nummber=chunk_number,
            is_correct=result_dict["is_correct"],
            reasoning=result_dict["reasoning"]
        )



    def process_query(self, query: str, source: str, chunk_number: int, q_number: int) -> str:
        search_results = self.retriever.search(query)
        print(f"✅ Found {len(search_results)} results for query: {query}\n")

        query_result = pd.DataFrame(columns=['eval', 'q_number', 'question', 'source', 'chunk_number', 'distance'])
        data = {'eval': 'correct', 'q_number': q_number, 'question': query, 'source': source,
                'chunk_number': chunk_number, 'distance': 0}
        row = pd.DataFrame(data, index=[0])
        query_result = pd.concat([query_result, row])

        for d in search_results:
            data = {'eval': 'suggestion', 'q_number': q_number, 'question': query, 'source': d['source'],
                    'chunk_number': d["number"], 'distance': d['_distance']}
            row = pd.DataFrame(data, index=[0])
            query_result = pd.concat([query_result, row])

        query_result.to_csv("data/out/eval/datastore_results_" + str(q_number) + ".csv", encoding="utf8")
        result_content = [d["content"] for d in search_results]
        response = self.generator.generate_response(query, result_content)

        return response

    def _evaluate(
            self, query: str, response: str, expected_answer: str
    ) -> str:
        context = f"""
        <question>\n{query}\n</question>
        <response>\n{response}\n</response>
        <expected_answer>\n{expected_answer}\n</expected_answer>
        """

        response_content = invoke_ai_json(system_message=SYSTEM_PROMPT, context=context, ret_object=AnswerEvaluation.model_json_schema())

        return response_content
