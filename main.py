import lancedb
import json
from wakepy.modes import keep
from src.impl.datastore import Datastore
from src.impl.generator import Generator
from src.impl.retriever import Retriever
from src.impl.evaluator import Evaluator
from src.impl.indexer import Indexer
from src.util import download_save_sentence_transformer, get_files_in_directory
from summarize import test_connection, summarize, get_from_pkl


do=4
DEFAULT_SOURCE_PATH = "data/pdfs/"

if do==1:
    "Generate chunks, summaries and questions and answers."
    with keep.running():
        document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
        print(document_paths)
        generator = Generator(document_paths)
        generator.generate_all_q_and_a()
        generator.generate_summaries()

if do==2:
    "Generate dataitems in datastore from the chunks, summaries and Q&As."
    with keep.running():
        indexer = Indexer()
        items = indexer.index()
        datastore = Datastore()
        datastore.reset()
        datastore.describe_table()
        datastore.add_items(items)
        datastore.head()
        print("max summary length")
        print(indexer.get_max_length_summary())
        print("max chunk length")
        print(indexer.get_max_length_chunk())

if do==3:
    datastore = Datastore()
    datastore.to_csv()

if do==4:
    "Get most fitting chunks for every question"
    with open("data/out/eval/qas.json", "r") as file:
        qas = json.load(file)

    print(qas)
    datastore = Datastore()
    datastore.reset()
    indexer = Indexer()
    items = indexer.index()
    datastore.add_items(items)
    retriever = Retriever(datastore=datastore)
    evaluator = Evaluator(retriever=retriever, q_and_as=qas)
    evaluator.evaluate()

if do==5:
    datastore = Datastore()
    print(datastore.get_number_of_records())












