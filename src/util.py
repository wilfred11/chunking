import json
import os
import pickle
from pathlib import Path
from typing import List
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from openai import OpenAI
from pydantic_core.core_schema import JsonSchema
from semchunk import semchunk
from sentence_transformers import SentenceTransformer

llm_url = os.environ.get("LMSTUDIO_URL")
llm_api_key = os.environ.get("LMSTUDIO_KEY")


class Record(BaseModel):
    question: str
    answer: str


class AnswerEvaluation(BaseModel):
    is_correct: str
    reasoning: str


class Response(BaseModel):
    generated: List[Record]


def download_save_sentence_transformer():
    model_name = os.environ.get("SENTENCE_TRANSFORMER")
    model_name_local = os.environ.get("LOCAL_SENTENCE_TRANSFORMER")
    dir_file = Path(model_name_local)
    if not dir_file.exists():
        emb_model = SentenceTransformer(model_name, trust_remote_code=True)
        emb_model.save(model_name_local)
        print("SentenceTransformer saved to " + model_name_local)
    else:
        print("SentenceTransformer already saved locally.")


def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    files_dirs = glob.glob(os.path.join(source_path, "*"))
    files = []
    for f_d in files_dirs:
        if Path(f_d).is_file():
            files.append(f_d)
    return files


def invoke_ai(system_message: str, context: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """
    client = OpenAI(base_url=llm_url, api_key=llm_api_key)
    response = client.chat.completions.create(
        #model="o4-mini",
        model="gemma-1.1-2b-it",
        messages=[
            {"role": "system", "content": system_message + context},
        ],
    )
    return response.choices[0].message.content


def invoke_ai_json(system_message: str, context: str, ret_object: JsonSchema) -> str:
    """
    Generic function to invoke an AI model given a system message and context. The OpenAI response is a json object.
    """
    client = OpenAI(base_url=llm_url, api_key=llm_api_key)
    response = client.chat.completions.create(
        model="gemma-1.1-2b-it",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "output_schema",
                "schema": ret_object
            }
        },
        messages=[
            {"role": "system", "content": system_message + context},
        ],
    )
    return response.choices[0].message.content


def process_query(self, query: str) -> str:
    search_results = self.retriever.search(query)
    print(f"✅ Found {len(search_results)} results for query: {query}\n")

    for i, result in enumerate(search_results):
        print(f"🔍 Result {i + 1}: {result}\n")

    response = self.response_generator.generate_response(query, search_results)
    return response


def semchunking(files):
    final_chunks = {}
    for file in files:
        print(file)
        loader = PyPDFLoader(file, mode="single")
        pages = [page.page_content for page in loader.load()]

        chunker = semchunk.chunkerify('cl100k_base', 256)
        chunks = chunker(pages[0].replace("\n", ""))
        chunk_number = 1
        for c in chunks:
            #print(c)
            print(len(c))
            key = (file, chunk_number)
            final_chunks[key] = c
            chunk_number = chunk_number + 1

    with open('data/out/pkl/final_chunks_sl.pkl', 'wb') as f:
        pickle.dump(final_chunks, f)
