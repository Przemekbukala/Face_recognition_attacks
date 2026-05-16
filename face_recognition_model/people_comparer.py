import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

from insightface.app import FaceAnalysis
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


device_id = -1  # GPU: 0, CPU: -1
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=device_id)


def get_embedding(img_path: str) -> Optional[np.ndarray]:
    """
    Extract face embedding using ArcFace (InsightFace).

    Parameters
    ----------
    img_path : str
        Path to image.

    Returns
    -------
    np.ndarray or None
        Face embedding vector.
    """
    p = Path(img_path)
    if not p.exists():
        print(f"Image not found: {img_path}")
        return None

    img = np.array(Image.open(img_path).convert("RGB"))

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

    emb1 = get_embedding("results/person4_grid_attack.jpg")
    emb2 = get_embedding("samples/person4.jpg")

    distance, result = compare_embeddings(emb1, emb2)

    print(f"Distance: {distance:.4f} - {result}")