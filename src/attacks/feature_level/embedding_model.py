from __future__ import annotations

_EMBEDDING_MODEL: dict = {}


def get_embedding_model(pretrained: str = "vggface2"):
    """
    PyTorch face-embedding model for feature-level attacks (FFM/FIM).
    InsightFace runs inference only and exposes no gradients, so perturbations are optimized on this model and then evaluated on InsightFace
    """
    if pretrained in _EMBEDDING_MODEL:
        return _EMBEDDING_MODEL[pretrained]

    import torch
    from facenet_pytorch import InceptionResnetV1

    device = torch.device("cpu")
    model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    _EMBEDDING_MODEL[pretrained] = (model, device)
    return model, device
