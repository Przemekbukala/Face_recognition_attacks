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
- **InsightFace models** download into `data/insightface/` inside the repo. 
- `facenet-pytorch` will download pretrained models on first run
- On CPU-only machines you may need to install a CPU-compatible `torch` build.

**LFW + benchmark** -  `benchmark.py`:

1. Download LFW via KaggleHub, then **copy** images and pair lists into `data/lfw/` in the repo.
2. Run all registered attacks on the dataset.
3. Write `results/benchmark_results.json`, `results/run_info.json`, and `results/benchmark_summary.txt`.

```bash
python3 benchmark.py                  # full benchmark
python3 benchmark.py --n-pairs 300    # quick subset
```
