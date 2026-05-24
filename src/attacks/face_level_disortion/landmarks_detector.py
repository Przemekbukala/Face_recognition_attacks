import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from insightface.app import FaceAnalysis


app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1)


def path_to_img(path: str) -> Optional[np.ndarray]:

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


def draw_landmarks(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    kps: (5,2) keypoints from InsightFace
    order: left_eye, right_eye, nose, mouth_left, mouth_right
    """
    out = img.copy()
    h, w, _ = out.shape

    for (x, y) in kps:
        x = int(x)
        y = int(y)
        if 0 <= x < w and 0 <= y < h:
            out[y-1:y+1, x-1:x+1] = [0, 255, 0]

    return out

def get_labeled_kps(kps: np.ndarray):
    """
    Returns:
    [((x,y), label), ...]
    """

    labels = [
        "left_eye",
        "right_eye",
        "nose",
        "mouth_left",
        "mouth_right"
    ]

    return [
        ((float(x), float(y)), label)
        for (x, y), label in zip(kps, labels)
    ]


if __name__ == "__main__":

    img = path_to_img("samples/person1.jpg")
    if img is None:
        exit()
    faces = app.get(img)

    if len(faces) == 0:
        print("No face detected")
        exit()
    face = faces[0]
    kps = face.kps
    img_out = draw_landmarks(img, kps)

    Image.fromarray(img_out).save(
        "results/face_level_disortion/landmarks.jpg"
    )
    labeled_kps = get_labeled_kps(kps)

    print(labeled_kps)