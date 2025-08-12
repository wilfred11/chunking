import os
from pathlib import Path
from typing import List
import glob

from sentence_transformers import SentenceTransformer


def download_save_sentence_transformer():
    model_name = os.environ.get("SENTENCE_TRANSFORMER")
    model_name_local = os.environ.get("LOCAL_SENTENCE_TRANSFORMER")
    dir_file = Path(model_name_local)
    if not dir_file.exists():
        emb_model = SentenceTransformer(model_name, trust_remote_code=True)
        emb_model.save(model_name_local)
        print("SentenceTransformer saved to " + model_name_local)
    else:
        print("SentenceTransformer already saved locally.")

def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    return glob.glob(os.path.join(source_path, "*"))