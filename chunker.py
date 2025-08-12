import os
from os.path import isfile, join
from pathlib import Path
import pyarrow as pa
import pandas as pd
from docling.datamodel.pipeline_options import PipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DocItemLabel
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker, DocChunk
from docling.chunking import HybridChunker
#https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257


PDF_DIR = "data/pdfs"

pdf_names = ["data/pdfs/sl_booklet.pdf", "data/pdfs/sl_service_guide.pdf"]


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
        print(chunk.page_content + "\n")


def hierarchical_chunk():
    converter = DocumentConverter()
    chunker = HierarchicalChunker()

    # Convert the input file to Docling Document
    doc = converter.convert(pdf_names[0]).document

    # Perform hierarchical chunking
    texts = [chunk.text for chunk in chunker.chunk(doc)]

    contexts = []
    for c in chunker.chunk(doc):
        contexts.append(chunker.contextualize(chunk=c))

    for i, (text, context) in enumerate(zip(texts, contexts)):
        print(f"Chunk {i + 1}:\n{text}\n{'-' * 50}")
        print(f"Context {i + 1}:\n{context}\n{'-' * 50}")


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


def text_table_chunk():
    output_dir_chunks = Path("data/out/generated/chunks")
    output_dir_tables = Path("data/out/generated/tables")
    #pipeline_options = PipelineOptions(do_table_structure=True)
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert("data/pdfs/research-methods/rm.pdf")
    output_dir_chunks.mkdir(parents=True, exist_ok=True)
    output_dir_tables.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem
    doc = conv_res.document
    tables = doc.tables
    print(tables)

    for table_ix, table in enumerate(tables):
        print("table.label")
        print(table.label)
        print(table.get_ref())
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = output_dir_tables / f"{doc_filename}-table-{table_ix + 1}.csv"
        table_df.to_csv(element_csv_filename)

        # Save the table as html
        element_html_filename = output_dir_tables / f"{doc_filename}-table-{table_ix + 1}.html"
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

    #print(doc.tables.count())


    #print(doc.tables)
    print(type(doc))

    chunker = HybridChunker(max_tokens=512, merge_peers=True)
    chunk_iter = chunker.chunk(dl_doc=doc)
    #print(sum(1 for _ in chunk_iter))
    chunk_table = {}
    for i, chunk in enumerate(chunk_iter):
        doc_chunk = DocChunk.model_validate(chunk)
        for it in doc_chunk.meta.doc_items:
            table_list = []
            if it.label == DocItemLabel.TABLE:
                print("it")
                print(it.get_ref())
                table_list.append(it.get_ref())

                """element_html_filename = output_dir_tables / f"{doc_filename}-table-{table_ix + 1}.html"
                with element_html_filename.open("w") as fp:
                    fp.write(chunk..export_to_html(doc=conv_res.document))"""

                print("chunk (w table)")
                print(chunk.text[:200])


        chunk_table.update(i, table_list)
    print(chunk_table)
            #else:
            #    print("chunk (no table)")
            #    print(chunk.text[:200])
    #    enriched_text = chunker.serialize(chunk=chunk)




"""def get_embedding_model():
    embedding_model = (
        get_registry()
        .get("sentence-transformers")
        .create(
            name="all-mpnet-base-v2", device="cpu"
        )
    )
    return embedding_model


em = get_embedding_model()


def get_embedding(text):
    embedding = (
        em.embedding_model.encode(text, convert_to_tensor=True)
        .cpu()
        .numpy()
    )
    return embedding


pa_schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("pdf_id", pa.string()),
        pa.field("pdf", pa.binary()),
        pa.field("context", pa.string()),
    ]
)


class Schema(LanceModel):
    vector: Vector(em.ndims()) = em.VectorField()
    pdf_id: str
    pdf: bytes
    context: str = em.SourceField()


def process_dataset(dataset):
    pdffiles = [f for f in os.listdir(PDF_DIR) if isfile(join(PDF_DIR, f))]

    for idx, (pdf_id, caption) in pdffiles:
        try:
            with open(os.path.join(PDF_DIR, image_id), "rb") as image:
                binary_image = image.read()

        except FileNotFoundError:
            print(f"image_id '{image_id}' not found in the folder, skipping.")
            continue

        pdf_id = pa.array([image_id], type=pa.string())
        pdf = pa.array([binary_image], type=pa.binary())
        context = pa.array([context], type=pa.string())

        # Ensure caption is a string when passed to get_embedding
        embedding = pa.array(
            [get_embedding(str(caption[0]))], type=pa.list_(pa.float32(), 768)
        )

        yield pa.RecordBatch.from_arrays(
            [embedding, pdf_id, pdf, caption],
            ["vector", "image_id", "image", "captions"],
        )
"""
