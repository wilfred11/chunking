import lancedb

from chunker import chunk, hierarchical_chunk, table_chunk
from vector import embed, get_annotations, get_embedding_model, Schema, process_dataset, create_vector_db

do=4
#https://www.youtube.com/watch?v=V58mPkLB95o
#https://python.langchain.com/docs/integrations/document_loaders/docling/

if do==1:
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