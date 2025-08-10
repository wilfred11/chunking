from lancedb.embeddings import get_registry, SentenceTransformerEmbeddings
import os
import io
import lancedb
import pandas as pd
import pyarrow as pa
from PIL import Image
from matplotlib import pyplot as plt
from lancedb.pydantic import LanceModel, Vector

IMAGE_DIR = "data/Images"
ANNOTATION_FILE = "data/captions.txt"


def get_embedding_model():
    embedding_model = (
        get_registry()
        .get("sentence-transformers")
        .create(
            name="all-mpnet-base-v2", device="cpu"
        )
    )
    return embedding_model

em=get_embedding_model()

def get_embedding(text):
    """Get embedding of the text from the embedding model"""
    embedding = (
        em.embedding_model.encode(text, convert_to_tensor=True)
        .cpu()
        .numpy()
    )
    return embedding


def embed():
    embedding_model = get_embedding_model()

    text1 = "king"
    text2 = "queen"
    text3 = "apple"

    embedding1 = get_embedding(text1, embedding_model)
    embedding2 = get_embedding(text2, embedding_model)
    embedding3 = get_embedding(text3, embedding_model)

    print(f"Similarity between `{text1}` and `{text2}`: {embedding1.dot(embedding2):.2f}")
    print(f"Similarity between `{text1}` and `{text3}`: {embedding1.dot(embedding3):.2f}")


def get_annotations():
    df = pd.read_csv(ANNOTATION_FILE)
    df = df.groupby("image")["caption"].apply(list).reset_index(name="caption")
    df["caption"] = df["caption"].apply(lambda x: " ".join(x))
    print(df.head())
    return df.head(50)



pa_schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("image_id", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("captions", pa.string()),
    ]
)


class Schema(LanceModel):
    vector: Vector(em.ndims()) = em.VectorField()
    image_id: str
    image: bytes
    captions: str = em.SourceField()

def process_dataset(dataset):
    for idx, (image_id, caption) in enumerate(dataset.values):
        try:
            with open(os.path.join(IMAGE_DIR, image_id), "rb") as image:
                binary_image = image.read()

        except FileNotFoundError:
            print(f"image_id '{image_id}' not found in the folder, skipping.")
            continue

        image_id = pa.array([image_id], type=pa.string())
        image = pa.array([binary_image], type=pa.binary())
        caption = pa.array([caption], type=pa.string())

        # Ensure caption is a string when passed to get_embedding
        embedding = pa.array(
            [get_embedding(str(caption[0]))], type=pa.list_(pa.float32(), 768)
        )

        yield pa.RecordBatch.from_arrays(
            [embedding, image_id, image, caption],
            ["vector", "image_id", "image", "captions"],
        )

def show_image(image):
    stream = io.BytesIO(image)
    plt.imshow(Image.open(stream))
    plt.axis("off")
    plt.show()

def create_vector_db(df):
    db = lancedb.connect("embedding_dataset")
    tbl = db.create_table("table", schema=Schema, mode="overwrite")
    tbl.add(process_dataset(df))

    query = "dog running through sand"

    hit_lists = tbl.search(query).metric("cosine").limit(2).to_list()

    for hit in hit_lists:
        show_image(hit["image"])


    query = "many people happy"
    sub_query = "many people"
    rep_query = "single man"

    query_emb = get_embedding(query)
    sub_query_emb = get_embedding(sub_query)
    rep_query_emb = get_embedding(rep_query)

    emb = query_emb - sub_query_emb + rep_query_emb

    hit_lists = tbl.search(emb).metric("cosine").limit(6).to_list()

    for hit in hit_lists:
        show_image(hit["image"])