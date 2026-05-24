import numpy as np
import landmarks_detector as ld
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

if __name__ == "__main__":
    img = ld.path_to_img("samples/person1.jpg")
    kps = ld.app.get(img)[0].kps
    img_eyes = occlude_eyes(img, kps)
    img_nose = occlude_nose(img, kps)
    img_mouth = occlude_mouth(img, kps)

    Image.fromarray(img_eyes).save("results/face_level_disortion/eyes_occluded.jpg")
    Image.fromarray(img_nose).save("results/face_level_disortion/nose_occluded.jpg")
    Image.fromarray(img_mouth).save("results/face_level_disortion/mouth_occluded.jpg")