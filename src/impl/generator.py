import pickle
from typing import List
from src.interface.base_generator import BaseGenerator
from src.util import invoke_ai, invoke_ai_json, recursive_character_chunking
from unstructured_chunking import semchunk, semchunking

#from src.util.invoke_ai import invoke_ai


SYSTEM_PROMPT = """
Use the provided context to provide a concise answer to the user's question.
If you cannot find the answer in the context, say so. Do not make up information.
"""

SUMMARY_PROMPT = """
"Summarize the following text in a commercial way. Focus on facts, ideas used. Add a fitting title to the summary. Be impersonal.
"""

QA_PROMPT = """
Use the provided context to generate one question and matching answer. The question should ask something meaningful about the context. And the context should contain information to answer the question.
 The question and its answer should be returned in json format. If no sensible question can be generated from the context, just return an empty json object.
"""


class Generator(BaseGenerator):
    def __init__(self, files : List[str]):
        semchunking(files)

        with open('data/out/pkl/final_chunks_sl.pkl', 'rb') as chunks:
            self.final_chunks: dict(str,str) = pickle.load(chunks)


    def generate_all_q_and_a(self):
        all_qa={}
        for i,((file,chunk_number),v) in enumerate(self.final_chunks.items()):
            print(v)
            qa=self.generate_q_and_a(v)
            print("qa")
            print(qa)
            all_qa[(file,chunk_number)]=qa
            #if i == 5:
            #    break

        with open('data/out/pkl/qa_sl.pkl', 'wb') as f:
            pickle.dump(all_qa, f)

    def generate_summaries(self):
        summaries = {}
        for i,((file,chunk_number),v) in enumerate(self.final_chunks.items()):
            summary=self.generate_summary(v)
            print("summary")
            print(summary)
            print(len(summary))
            summaries[(file, chunk_number)]=summary
            #if i == 1:
            #    break
        with open('data/out/pkl/chunk_summaries_sl.pkl', 'wb') as f:
            pickle.dump(summaries, f)


    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using OpenAI's chat completion."""
        # Combine context into a single string
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)

    def generate_summary(self, context: str):
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
        )
        return invoke_ai(system_message=SUMMARY_PROMPT, user_message=user_message)

    def generate_q_and_a(self, context: str):
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
        )

        return invoke_ai_json(system_message=QA_PROMPT, user_message=user_message)

    def generate_q_and_a1(self, context: str):
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
        )

        return invoke_ai_json(system_message=QA_PROMPT, user_message=user_message)

