from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import pandas as pd

from src.interface.base_evaluator import BaseEvaluator, EvaluationResult
from src.interface.base_generator import BaseGenerator
from src.interface.base_retriever import BaseRetriever
from src.util import invoke_ai_json

#from src.util.invoke_ai import invoke_ai
#from src.util.extract_xml import extract_xml_tag

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
The question will be provided in <question>...</question> tags.
The response will be provided in <response>...</response> tags.
The expected answer will be provided in <expected_answer>...</expected_answer> tags.

The response doesn't have to exactly match all the words/context the expected answer. It just needs to be right about
the answer to the actual question itself.

Evaluate whether the response is correct or not, and return your reasoning in <reasoning>...</reasoning> tags.
Then return the result in <result>...</result> tags — either as 'true' or 'false'.
"""


class Evaluator(BaseEvaluator):
    def __init__(self, retriever: BaseRetriever, q_and_as ):
        self.retriever = retriever
        self.q_and_as=q_and_as

    def evaluate(self):
        questions = [item["question"] for item in self.q_and_as]
        expected_answers = [item["answer"] for item in self.q_and_as]
        files = [item["file"] for item in self.q_and_as]
        chunk_numbers = [item["chunk_number"] for item in self.q_and_as]
        q_numbers=range(1,len(files)+1)
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

        """for i, result in enumerate(results):
            result_emoji = "✅" if result.is_correct else "❌"
            print(f"{result_emoji} Q {i+1}: {result.question}: \n")
            print(f"Response: {result.response}\n")
            print(f"Expected Answer: {result.expected_answer}\n")
            print(f"Reasoning: {result.reasoning}\n")
            print("--------------------------------")"""

        #number_correct = sum(result.is_correct for result in results)
        #print(f"✨ Total Score: {number_correct}/{len(results)}")
        return results


    def _evaluate_single_question(
        self, question: str, expected_answer: str, source:str, chunk_number:int, q_number:int
    ) -> EvaluationResult:
        # Evaluate a single question/answer pair.
        response = self.process_query(question,source, chunk_number,q_number)
        return None
        #return self.evaluator.evaluate(question, response, expected_answer)

    def process_query(self, query: str, source: str, chunk_number: int, q_number: int) -> str:
        search_results = self.retriever.search(query)
        print(f"✅ Found {len(search_results)} results for query: {query}\n")

        query_result = pd.DataFrame(columns=['eval','q_number','question', 'source', 'chunk_number'])
        data = {'eval':'correct','q_number': q_number, 'question': query,'source': source, 'chunk_number': chunk_number}
        row = pd.DataFrame(data, index=[0])
        query_result = pd.concat([query_result, row])

        for d in search_results:
            data = {'eval': 'suggestion','q_number':q_number, 'question':query,'source':d['source'], 'chunk_number':d["number"]}
            row=pd.DataFrame(data, index=[0])
            query_result=pd.concat([query_result, row])

        query_result.to_csv("data/out/eval/datastore_results_"+str(q_number)+".csv", encoding="utf8")

        #response = self.generator.generate_response(query, search_results)
        return None

    def _evaluate(
        self, query: str, response: str, expected_answer: str
    ) -> EvaluationResult:

        user_prompt = f"""
        <question>\n{query}\n</question>
        <response>\n{response}\n</response>
        <expected_answer>\n{expected_answer}\n</expected_answer>
        """

        response_content = invoke_ai_json(
            system_message=SYSTEM_PROMPT, user_message=user_prompt
        )

        #reasoning = extract_xml_tag(response_content, "reasoning")
        #result = extract_xml_tag(response_content, "result")
        print(response_content)

        """if result is not None:
            is_correct = result.lower() == "true"
        else:
            is_correct = False
            reasoning = f"No result found: ({response_content})"""""

        return EvaluationResult(
            question=query,
            #response=response,
            expected_answer=expected_answer,
            #is_correct=is_correct,
            #reasoning=reasoning,
        )
