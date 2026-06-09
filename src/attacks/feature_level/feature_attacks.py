"""
Feature-level adversarial attacks FFM / FIM.

    FFM = Feature Fast Attack Method       -- single-step 
    FIM = Feature Iterative Attack Method   -- iterative 
    """

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .embedding_model import get_embedding_model
except ImportError:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from src.attacks.feature_level.embedding_model import get_embedding_model


def _preprocess(x, input_size: int):
    """Resize a (1,3,H,W) [0,255] tensor to the embedding-model input + facenet prewhitening."""
    x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return (x - 127.5) / 128.0


def _face_bbox(img: np.ndarray, margin: float):
    """Detect the largest face bbox with InsightFace and expand it by margin."""
    from face_recognition_model.people_comparer import app

    faces = app.get(img)
    if not faces:
        return None

    face = max(faces,key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),)
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    h, w = img.shape[:2]

    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(w, int(np.ceil(x2 + pad_x)))
    y2 = min(h, int(np.ceil(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _attack_mask(img: np.ndarray, face_only: bool, face_margin: float, device):
    """Return (1,3,H,W) mask for the perturbation update."""
    import torch

    h, w = img.shape[:2]
    mask = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
    if not face_only:
        mask.fill_(1.0)
        return mask

    bbox = _face_bbox(img, face_margin)
    if bbox is None:
        mask.fill_(1.0)
        return mask

    x1, y1, x2, y2 = bbox
    mask[:, :, y1:y2, x1:x2] = 1.0
    return mask


def _feature_attack(img: np.ndarray, epsilon: float, steps: int, alpha: float, pretrained: str,input_size: int, face_only: bool, face_margin: float):

    model, device = get_embedding_model(pretrained)

    arr = np.asarray(img, dtype=np.float32) 
    x_orig = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) 
    mask = _attack_mask(img, face_only, face_margin, device)

    with torch.no_grad():
        e0 = F.normalize(model(_preprocess(x_orig, input_size)), dim=1) 

    x_adv = x_orig.clone()
    for _ in range(max(1, steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        emb = F.normalize(model(_preprocess(x_adv, input_size)), dim=1)
        similarity = (emb * e0).sum()  
        grad = torch.autograd.grad(similarity, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign() * mask
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            x_adv = x_adv.clamp(0.0, 255.0)

    out = x_adv.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def fim_attack(img: np.ndarray, epsilon: float = 8.0, steps: int = 10, alpha: float = 2.0, pretrained: str = "vggface2", input_size: int = 160, face_only: bool = True, face_margin: float = 0.25):
    """Iterative feature-level dodging attack (FIM)."""
    return _feature_attack(img, epsilon, steps, alpha, pretrained, input_size, face_only, face_margin)


def ffm_attack(img: np.ndarray, epsilon: float = 8.0, pretrained: str = "vggface2", input_size: int = 160, face_only: bool = True, face_margin: float = 0.25):
    """Single-step FFM = Feature Fast Attack Method"""
    return _feature_attack(img, epsilon, steps=1, alpha=epsilon, pretrained=pretrained, input_size=input_size, face_only=face_only, face_margin=face_margin)


if __name__ == "__main__":
    from PIL import Image

    orig = np.asarray(Image.open("samples/person1.jpg").convert("RGB"), dtype=np.uint8)
    epsilon = 12.0
    adv = ffm_attack(orig, epsilon=epsilon, face_only=True)

    diff = np.abs(orig.astype(np.float32) - adv.astype(np.float32))
    mag = np.clip(diff * 8.0, 0, 255).astype(np.uint8)

    out_dir = Path("results/feature_level")
    out_dir.mkdir(parents=True, exist_ok=True)
    p_przed = out_dir / "ffm_przed.png"
    p_po = out_dir / "ffm_po.png"
    p_roznica = out_dir / "ffm_roznica_x8.png"
    Image.fromarray(orig).save(p_przed)
    Image.fromarray(adv).save(p_po)
    Image.fromarray(mag).save(p_roznica)
    print(f"Zapisano: {p_przed}")
    print(f"Zapisano: {p_po}")
    print(f"Zapisano: {p_roznica}")
