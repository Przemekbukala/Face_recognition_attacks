from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from pathlib import Path
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    p = Path(img_path)
    if not p.exists():
        print(f"Image not found: {img_path}")
        return None
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Image not found: {img_path}")
        return None

    face = mtcnn(img)

    if face is None:
        print(f"No face detected in: {img_path}")
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face)

    return embedding

repo_root = Path(__file__).resolve().parent.parent
emb1 = get_embedding(str(repo_root / "person1.jpg"))
emb2 = get_embedding(str(repo_root / "person3.jpg"))

if emb1 is None or emb2 is None:
    print("Unable to compute embeddings for one or both images. Exiting.")
    sys.exit(1)

distance = torch.dist(emb1, emb2).item()

print("Distance:", distance)

if distance < 1.0:
    print("Same person")
else:
    print("Different person")
