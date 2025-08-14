from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pickle
from langchain_openai.embeddings import OpenAIEmbeddings
from semantic_chunker import SemanticChunker
from semantic_chunker.integrations.langchain import SemanticChunkerSplitter
from semchunk import semchunk


def recursive_character_chunking():
    # need table extracting , camelot
    loader = PyPDFLoader(file_path="data/pdfs/qa/ICSE23.pdf", mode="single")
    pages = loader.load()

    recursive_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1024,
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
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256,
        chunk_overlap=0,
    )

    recursive_text_splitter1 = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=768,
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

    #token_text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=256, chunk_overlap=50)

    text_splitter_chunks = recursive_text_splitter.split_documents(pages)
    final_chunks = {}
    numbering=[]
    count_m=0
    count_n=0
    for c in text_splitter_chunks:
        new_text_splitter_chunks= recursive_text_splitter1.split_text(c.page_content)
        count_m = count_m+1
        for nc in new_text_splitter_chunks:
            count_n= count_n +1
            #print(nc)
            print(len(nc))
            #final_chunks.append(nc)
            key=str(count_m)+"."+str(count_n)
            final_chunks[key] = nc
            numbering.append(key)

    with open('data/out/pkl/final_chunks1.pkl', 'wb') as f:
        pickle.dump(final_chunks, f)
    with open('data/out/pkl/numbering1.pkl', 'wb') as f:
        pickle.dump(numbering, f)

from langchain_community.document_loaders import PyPDFLoader

def test():
    pdf_path = "data/pdfs/sl_booklet.pdf"
    loader = PyPDFLoader(pdf_path, mode="single")

    #    Load all pages and extract text content
    pages = [page.page_content for page in loader.load()]

    print(pages)
    pages

    chunker = semchunk.chunkerify('cl100k_base', 512)
    chunks=chunker(pages[0].replace("\n",""))
    for c in chunks:
        print(c)
        print(len(c))


def semchunking(files):
    final_chunks={}
    for file in files:
        print(file)
        loader = PyPDFLoader(file, mode="single")
        pages = [page.page_content for page in loader.load()]

        chunker = semchunk.chunkerify('cl100k_base', 256)
        chunks=chunker(pages[0].replace("\n",""))
        chunk_number = 1
        for c in chunks:
            #print(c)
            print(len(c))
            key = (file, chunk_number)
            final_chunks[key] = c
            chunk_number=chunk_number+1

    with open('data/out/pkl/final_chunks_sl.pkl', 'wb') as f:
        pickle.dump(final_chunks, f)



"""def test1():
    pdf_path = "data/pdfs/sl_booklet.pdf"
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    #loader.load()
    #    Load all pages and extract text content
    #pages = [page.page_content for page in loader.lazy_load()]
    #print(pages)
    chunker = SemanticChunkerSplitter(max_tokens=512)
    chunks = chunker.split_documents(result.document)
    for c in chunks:
        print(c)"""








