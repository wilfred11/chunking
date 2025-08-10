from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker

pdf_names=["pdfs/sl_booklet.pdf", "pdfs/sl_service_guide.pdf"]

def chunk():
    loader = PyPDFLoader(pdf_names[1])
    documents = loader.load()
    print(len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
    )

    naive_chunks = text_splitter.split_documents(documents)
    for chunk in naive_chunks:
      print(chunk.page_content+ "\n")

def hierarchical_chunk():
    converter = DocumentConverter()
    chunker = HierarchicalChunker()

    # Convert the input file to Docling Document
    doc = converter.convert(pdf_names[1]).document

    # Perform hierarchical chunking
    texts = [chunk.text for chunk in chunker.chunk(doc)]

    for i, text in enumerate(texts):
        print(f"Chunk {i + 1}:\n{text}\n{'-' * 50}")

def table_chunk():
    output_dir = Path("generated")
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(pdf_names[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        table_df.to_csv(element_csv_filename)

        # Save the table as html
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

