import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

from insightface.app import FaceAnalysis
import logging
import warnings
import os
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)


device_id = -1

devnull = open(os.devnull, "w")
old_stdout = sys.stdout
sys.stdout = devnull

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=device_id)

sys.stdout = old_stdout
devnull.close()

def path_to_img(path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray or None
        Image in HWC format or None if loading fails.
    """
    p = Path(path)
    if not p.exists():
        print(f"Image not found: {path}")
        return None

    try:
        img = np.array(Image.open(p).convert("RGB"))
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract face embedding using ArcFace (InsightFace).

    Parameters
    ----------
    img : np.ndarray
        Image in HWC format.

    Returns
    -------
    np.ndarray or None
        Face embedding vector.
    """
    faces = app.get(img)

    if len(faces) == 0:
        print(f"No face detected: {img_path}")
        return None

    return faces[0].embedding


def compare_embeddings(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = 0.45
) -> Tuple[float, str]:
    """
    Compare two ArcFace embeddings using cosine distance.
    """

    if emb1 is None or emb2 is None:
        return float("inf"), "Invalid embeddings"

    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    cosine_distance = 1 - np.dot(emb1, emb2)

    if cosine_distance < threshold:
        result = "Same person"
    else:
        result = "Different person"

    return float(cosine_distance), result


if __name__ == "__main__":

    emb1 = get_embedding(path_to_img("results/person3_grid_attack.jpg"))
    emb2 = get_embedding(path_to_img("samples/person3.jpg"))

    distance, result = compare_embeddings(emb1, emb2)

    print(f"Distance: {distance:.4f} - {result}")