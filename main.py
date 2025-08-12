import lancedb
from wakepy.modes import keep
from chunker import chunk, hierarchical_chunk, table_chunk, text_table_chunk
from src.impl.datastore import Datastore
from src.impl.indexer import Indexer
from src.util import download_save_sentence_transformer, get_files_in_directory
from summarize import test_connection, summarize, get_from_pkl
from unstructured_chunking import  recursive_character_chunking
from vector import embed, get_annotations, get_embedding_model, Schema, process_dataset, create_vector_db

do=8
DEFAULT_SOURCE_PATH = "data/pdfs/qa"


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



if do==8:
    with keep.running():
        recursive_character_chunking()
        #test_connection()
        summarize()

if do==7:
    with keep.running():
        document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)
        print(document_paths)
        download_save_sentence_transformer()
        get_from_pkl()
        datastore = Datastore()
        indexer = Indexer()
        items = indexer.index("ICSE.pdf")
        print("max summary length")
        print(indexer.get_max_length_summary())
        print("max chunk length")
        print(indexer.get_max_length_chunk())










