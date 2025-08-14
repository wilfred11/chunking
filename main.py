import lancedb
from wakepy.modes import keep
from src.impl.datastore import Datastore
from src.impl.generator import Generator
from src.impl.indexer import Indexer
from src.util import download_save_sentence_transformer, get_files_in_directory
from summarize import test_connection, summarize, get_from_pkl
from unstructured_chunking import recursive_character_chunking, test
from vector import embed, get_annotations, get_embedding_model, Schema, process_dataset, create_vector_db

do = 8
DEFAULT_SOURCE_PATH = "data/pdfs/"

#https://www.youtube.com/watch?v=V58mPkLB95o
#https://python.langchain.com/docs/integrations/document_loaders/docling/

"""if do==1:
    hierarchical_chunk()
if do==2:
    chunk()
if do==3:
    table_chunk()

if do==4:
    #m=get_embedding_model()
    #print(type(m))
    df=get_annotations()
    create_vector_db(df)

    #embed()

if do==5:
    text_table_chunk()"""

if do == 1:
    with keep.running():
        document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)

        print(document_paths)
        generator = Generator(document_paths)
        generator.generate_all_q_and_a()

if do == 8:
    with keep.running():
        document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
        generator = Generator(document_paths)
        generator.generate_summaries()
        generator.generate_all_q_and_a()
        #test_connection()
        #summarize()

if do == 7:
    with keep.running():
        document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
        print(document_paths)
        download_save_sentence_transformer()
        #get_from_pkl()
        datastore = Datastore()
        datastore.reset()
        indexer = Indexer()
        items = indexer.index("ICSE.pdf")
        print("max summary length")
        print(indexer.get_max_length_summary())
        print("max chunk length")
        print(indexer.get_max_length_chunk())

        datastore.add_items(items)

if do == 6:
    datastore = Datastore()
    datastore.reset()
    datastore.describe_table()
    indexer = Indexer()
    items = indexer.index("ICSE.pdf")
    datastore.add_items(items)

if do == 9:
    datastore = Datastore()
    datastore.head()

if do == 10:
    test()
