from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pickle



def recursive_character_chunking():
    # need table extracting , camelot
    loader = PyPDFLoader(file_path="data/pdfs/qa/ICSE23.pdf", mode="single")
    pages = loader.load()

    recursive_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=256,
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

    #token_text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=256, chunk_overlap=50)

    text_splitter_chunks = recursive_text_splitter.split_documents(pages)
    final_chunks = {}
    numbering=[]
    count_m=0
    count_n=0
    for c in text_splitter_chunks:
        new_text_splitter_chunks= text_splitter.split_text(c.page_content)
        count_m = count_m+1
        for nc in new_text_splitter_chunks:
            count_n= count_n +1
            #print(nc)
            print(len(nc))
            #final_chunks.append(nc)
            key=str(count_m)+"."+str(count_n)
            final_chunks[key] = nc
            numbering.append(key)

    with open('data/out/pkl/final_chunks.pkl', 'wb') as f:
        pickle.dump(final_chunks, f)
    with open('data/out/pkl/numbering.pkl', 'wb') as f:
        pickle.dump(numbering, f)






