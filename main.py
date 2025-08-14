import lancedb
from wakepy.modes import keep
from src.impl.datastore import Datastore
from src.impl.generator import Generator
from src.impl.indexer import Indexer
from src.util import download_save_sentence_transformer, get_files_in_directory
from summarize import test_connection, summarize, get_from_pkl
from unstructured_chunking import recursive_character_chunking, test
from vector import embed, get_annotations, get_embedding_model, Schema, process_dataset, create_vector_db

do=3
DEFAULT_SOURCE_PATH = "data/pdfs/"

#https://www.youtube.com/watch?v=V58mPkLB95o
#https://python.langchain.com/docs/integrations/document_loaders/docling/

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
    """do vector search using questions, rerank"""

if do==10:
    test()












