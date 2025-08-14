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
from sentence_transformers import SentenceTransformer

class Record(BaseModel):
    question: str
    answer: str

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
    files_dirs=glob.glob(os.path.join(source_path, "*"))
    files=[]
    for f_d in files_dirs:
        if Path(f_d).is_file():
            files.append(f_d)
    return files

def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """
    client = OpenAI(base_url="http://192.168.178.66:1234/v1", api_key="lm-studio")

    #client = OpenAI()  # Insert the API key here, or use env variable $OPENAI_API_KEY.
    response = client.chat.completions.create(
        #model="o4-mini",
        model="gemma-1.1-2b-it",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def invoke_ai_json(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """
    client = OpenAI(base_url="http://192.168.178.66:1234/v1", api_key="lm-studio")

    #client = OpenAI()  # Insert the API key here, or use env variable $OPENAI_API_KEY.
    response = client.chat.completions.create(
        #model="o4-mini",
        model="gemma-1.1-2b-it",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content



"""def invoke_ai_json(system_message: str, user_message: str) -> dict:
    #mistralai/Mistral-Nemo-Instruct-2407
    client = OpenAI(base_url="http://192.168.178.25:1234/v1", api_key="lm-studio")

    stream = client.chat.completions.create(
        # model="o4-mini",
        model="gemma-1.1-2b-it",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        format=Response.model_json_schema(),
        stream=True
    )



    data = ""
    for x in stream:
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None:
            print(delta)
            data += delta
    return json.loads(data)
"""

def recursive_character_chunking(files):
    # need table extracting , camelot

    for file in files:
        if Path(file).is_file():
            loader = PyPDFLoader(file_path=file, mode="single")
            pages = loader.load()

            recursive_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=384,
                chunk_overlap=0,
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ".",
                    ",",
                    "\u200b",  # Zero-width space
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ]
            )

            text_splitter_chunks = recursive_text_splitter.split_documents(pages)
            final_chunks = {}
            count_m=0
            for chunk in text_splitter_chunks:
                count_m = count_m+1
                key = (file, count_m)
                cleaned_chunk = chunk.page_content.replace("\n", "")
                final_chunks[key] = cleaned_chunk
                print(len(cleaned_chunk))
                print(key)

    with open('data/out/pkl/final_chunks.pkl', 'wb') as f:
        pickle.dump(final_chunks, f)
