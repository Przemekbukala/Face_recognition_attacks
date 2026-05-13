# Face_recognition_attacks
Jakub Ledwoń, Przemysław Bukała, Artur Zamorowski

Short instructions to setup the environment.

1. Clone the repository and change into it:

```bash
git clone https://github.com/Przemekbukala/Face_recognition_attacks.git
cd Face_recognition_attacks
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes and troubleshooting:
- `facenet-pytorch` will download pretrained models on first run
- On CPU-only machines you may need to install a CPU-compatible `torch` build.

