import numpy as np

from face_recognition_model.people_comparer import get_embedding, compare_embeddings
from . import landmarks_detector as ld
from PIL import Image

def occlude_eyes(
    img: np.ndarray,
    kps: np.ndarray,
    padding_x: int = 25,
    padding_y: int = 20
) -> np.ndarray:
    """
    Adds black rectangle covering both eyes.

    kps order:
    [left_eye, right_eye, nose, mouth_left, mouth_right]
    """

    out = img.copy()
    h, w, _ = out.shape

    left_eye = kps[0]
    right_eye = kps[1]

    x1 = int(min(left_eye[0], right_eye[0]) - padding_x)
    x2 = int(max(left_eye[0], right_eye[0]) + padding_x)

    y_center = int((left_eye[1] + right_eye[1]) / 2)

    y1 = y_center - padding_y
    y2 = y_center + padding_y

    x1 = max(0, x1)
    x2 = min(w, x2)
    y1 = max(0, y1)
    y2 = min(h, y2)

    out[y1:y2, x1:x2] = [0, 0, 0]

    return out


def occlude_nose(
    img: np.ndarray,
    kps: np.ndarray,
    size_x: int = 30,
    size_y: int = 40
) -> np.ndarray:
    """
    Adds black rectangle covering nose.
    """

    out = img.copy()
    h, w, _ = out.shape

    nose = kps[2]

    x = int(nose[0])
    y = int(nose[1])

    x1 = max(0, x - size_x)
    x2 = min(w, x + size_x)

    y1 = max(0, y - size_y)
    y2 = min(h, y + size_y)

    out[y1:y2, x1:x2] = [0, 0, 0]

    return out


def occlude_mouth(
    img: np.ndarray,
    kps: np.ndarray,
    padding_x: int = 50,
    padding_y: int = 50
) -> np.ndarray:
    """
    Adds black rectangle covering mouth area.
    """

    out = img.copy()
    h, w, _ = out.shape

    mouth_left = kps[3]
    mouth_right = kps[4]

    x1 = int(min(mouth_left[0], mouth_right[0]) - padding_x)
    x2 = int(max(mouth_left[0], mouth_right[0]) + padding_x)

    y_center = int((mouth_left[1] + mouth_right[1]) / 2)

    y1 = y_center - padding_y
    y2 = y_center + padding_y

    x1 = max(0, x1)
    x2 = min(w, x2)

    y1 = max(0, y1)
    y2 = min(h, y2)

    out[y1:y2, x1:x2] = [0, 0, 0]

    return out

def apply_occlusion(
    attack_name: str,
    img: np.ndarray,
    padding_x: int = 25,
    padding_y: int = 25
) -> np.ndarray:
    """
    Apply selected face occlusion attack.

    Parameters
    ----------
    attack_name : str
        Name of attack:
        - eyes
        - nose
        - mouth

    img : np.ndarray
        Input image.

    kps : np.ndarray
        Facial landmarks.

    x : int
        X-axis padding for occlusion.

    y : int
        Y-axis padding for occlusion.

    Returns
    -------
    np.ndarray
        Attacked image.
    """
    kps = ld.app.get(img)[0].kps
    if attack_name == "eyes":
        attack_fn = occlude_eyes
    elif attack_name == "nose":
        attack_fn = occlude_nose
    elif attack_name == "mouth":
        attack_fn = occlude_mouth
    return attack_fn(img, kps, padding_x, padding_y)

if __name__ == "__main__":
    img = ld.path_to_img("samples/person1.jpg")
    img_eyes = apply_occlusion("eyes", img)
    img_nose = apply_occlusion("nose", img)
    img_mouth = apply_occlusion("mouth", img)

    Image.fromarray(img_eyes).save("results/face_level_disortion/eyes_occluded.jpg")
    Image.fromarray(img_nose).save("results/face_level_disortion/nose_occluded.jpg")
    Image.fromarray(img_mouth).save("results/face_level_disortion/mouth_occluded.jpg")
    emb_img = get_embedding(img)
    emb_img_eyes = get_embedding(img_eyes)
    emb_img_nose = get_embedding(img_nose)
    emb_img_mouth = get_embedding(img_mouth)
    comparison_eyes = compare_embeddings(get_embedding(ld.path_to_img("samples/person2.jpg")), emb_img_eyes)
    comparison_nose = compare_embeddings(emb_img, emb_img_nose)
    comparison_mouth = compare_embeddings(emb_img, emb_img_mouth)
    print(f"Eyes occlusion similarity:  {comparison_eyes}")
    print(f"Nose occlusion similarity:  {comparison_nose}")
    print(f"Mouth occlusion similarity:  {comparison_mouth}")
