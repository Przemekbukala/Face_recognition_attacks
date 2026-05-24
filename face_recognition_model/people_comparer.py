import numpy as np
from PIL import Image
from pathlib import Path
from typing import Iterable, Optional, Tuple

from insightface.app import FaceAnalysis
import os
import sys
import argparse

_REPO_ROOT = Path(__file__).resolve().parent.parent
INSIGHTFACE_ROOT = _REPO_ROOT / "data" / "insightface"
INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)

device_id = -1  # GPU: 0, CPU: -1

devnull = open(os.devnull, "w")
old_stdout = sys.stdout
sys.stdout = devnull

app = FaceAnalysis(
    name="buffalo_l",
    root=str(INSIGHTFACE_ROOT),
    providers=["CPUExecutionProvider"],
)

app.prepare(ctx_id=device_id)

sys.stdout = old_stdout
devnull.close()


def path_to_img(path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    """
    p = Path(path)

    if not p.exists():
        print(f"Image not found: {path}")
        return None

    try:
        img = np.array(Image.open(p).convert("RGB"))
        return img

    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def get_embedding(img: np.ndarray, *, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Extract face embedding using ArcFace (InsightFace).

    Parameters
    ----------
    img : np.ndarray
        Image in HWC format.
    verbose : bool
        If True, print when no face is detected.
    """
    faces = app.get(img)

    if len(faces) == 0:
        if verbose:
            print("No face detected")
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

    if len(sys.argv) != 3:
        print("Usage:")
        print("python compare_faces.py <image1> <image2>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    img1 = path_to_img(image1_path)
    img2 = path_to_img(image2_path)

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    distance, result = compare_embeddings(emb1, emb2)

    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Distance: {distance:.4f}")
    print(f"Result: {result}")