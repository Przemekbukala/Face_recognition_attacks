"""
LFW Face Recognition Attack Benchmark
Usage:
    python benchmark.py                # all pairs
    python benchmark.py --n-pairs 300  # quick test
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.params_loader import get_config
from src.attacks.image_lvl_grid_based_occlusion import image_level_grid_occlusion
from src.attacks.face_level_disortion.landmark_occlusion import apply_occlusion
from src.lfw_dataset import LFWDataset
from face_recognition_model.people_comparer import get_embedding, compare_embeddings

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ATTACK_FUNCTIONS = {
    "grid_occlusion": lambda img, **p: image_level_grid_occlusion(img, **p),

    "face_level_occlusion_eyes": lambda img, **p: apply_occlusion(
        "eyes",
        img,
        **p
    ),

    "face_level_occlusion_nose": lambda img, **p: apply_occlusion(
        "nose",
        img,
        **p
    ),

    "face_level_occlusion_mouth": lambda img, **p: apply_occlusion(
        "mouth",
        img,
        **p
    ),
}
config = get_config()

ATTACKS = []

for a in config["attacks"]:
    name = a["name"]
    params = a.get("parameters", {})
    if name not in ATTACK_FUNCTIONS:
        print(f"[WARN] Skipping unknown attack: {name}")
        continue
    fn = ATTACK_FUNCTIONS[name]

    ATTACKS.append((name, lambda img, fn=fn, p=params: fn(img, **p)))

def determine_threshold(pairs, percentile=95):
    """Determine distance threshold based on a percentile of distances between different people pairs.
    Parameters
    ----------
    pairs : list
        List of image pairs.
    percentile : int, optional
        Percentile to use for threshold determination (default is 95).

    Returns
    -------
    float
        Determined threshold.
    """
    distances_same = []
    distances_different = []

    for pair in pairs:
        img1, img2 = pair.load_images()
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        dist, label = compare_embeddings(emb1, emb2, threshold=0.5)  # Use a high threshold to get raw distances

        if label == "Same person":
            distances_same.append(dist)
        elif label == "Different people":
            distances_different.append(dist)

    threshold = np.percentile(distances_different, percentile)
    print(f"Determined threshold at {percentile} percentile: {threshold:.4f}")
    return threshold
#  Benchmark 

def run_benchmark(pairs, threshold: float) -> dict:
    all_results = {}

    for attack_name, attack_fn in ATTACKS:
        print(f"\n{'='*60}")
        print(f"  Attack: {attack_name}  |  pairs: {len(pairs)}  |  threshold: {threshold}")
        print(f"{'='*60}")

        rows = []
        n_skipped = 0
        t0 = time.time()

        for pair in tqdm(pairs, desc=attack_name, unit="pair"):
            img1, img2 = pair.load_images()

            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)
            dist_before, label_before = compare_embeddings(emb1, emb2, threshold)

            if label_before == "Invalid embeddings":
                n_skipped += 1
                continue

            img1_attacked = attack_fn(img1)

            emb1_attacked = get_embedding(img1_attacked)
            dist_after, label_after = compare_embeddings(emb1_attacked, emb2, threshold)

            if label_after == "Invalid embeddings":
                n_skipped += 1
                continue

            fooled = label_after != label_before
            tqdm.write(f"  {'[FOOLED]  ' if fooled else '[NO CHANGE]'}  "
                       f"{pair.img1_path.stem} vs {pair.img2_path.stem}  "
                       f"{dist_before:.4f} -> {dist_after:.4f}  ({label_after})")

            rows.append({
                "img1": pair.img1_path.stem,
                "img2": pair.img2_path.stem,
                "distance_before": round(dist_before, 6),
                "label_before": label_before,
                "distance_after": round(dist_after, 6),
                "label_after": label_after,
                "fooled": fooled,
            })

        elapsed = time.time() - t0
        evaluated = len(rows)

        def avg(lst, key):
            return round(float(np.mean([r[key] for r in lst])), 4) if lst else float("nan")

        fooled_pct = round(sum(1 for r in rows if r["fooled"]) / evaluated * 100, 1) if evaluated else float("nan")

        print(f"\n{'─'*60}")
        print(f"  Evaluated: {evaluated}   Skipped: {n_skipped}   Time: {elapsed:.1f}s")
        print(f"  Avg distance before attack : {avg(rows, 'distance_before'):.4f}")
        print(f"  Avg distance after attack  : {avg(rows, 'distance_after'):.4f}")
        print(f"  Attack fooled model        : {fooled_pct:.1f}%  (Same -> Different person)")

        all_results[attack_name] = {
            "n_evaluated": evaluated,
            "n_skipped": n_skipped,
            "elapsed_seconds": round(elapsed, 1),
            "threshold": threshold,
            "avg_dist_before": avg(rows, "distance_before"),
            "avg_dist_after": avg(rows, "distance_after"),
            "fooled_pct": fooled_pct,
            "pairs": rows,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=-1,
                        help="Number of pairs (-1 = all)")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\nLoading LFW dataset\n")
    ds = LFWDataset(auto_download=True)
    print(f"    People: {ds.n_people}  |  Images: {ds.n_images}\n")

    n = args.n_pairs
    if n == -1:
        pairs = ds.get_pairs(n=-1, only_same=True, seed=args.seed)
    else:
        pairs = ds.get_pairs(n=n, only_same=True, seed=args.seed)

    print(f"Starting benchmark: {len(pairs)} pairs, attacks: {list(ATTACKS)}\n")
    thresh = determine_threshold(pairs, percentile=95)
    print(f"Using threshold: {thresh:.4f}\n")
    results = run_benchmark(pairs, threshold=thresh)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    txt_path = RESULTS_DIR / f"summary_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Benchmark  {ts}\n")
        f.write(f"Pairs: {len(pairs)}  Threshold: {args.threshold}\n\n")
        for attack_name, r in results.items():
            f.write(f"Attack: {attack_name}\n")
            f.write(f"  Evaluated: {r['n_evaluated']}  Skipped: {r['n_skipped']}  Time: {r['elapsed_seconds']}s\n")
            f.write(f"  Avg distance before: {r['avg_dist_before']}\n")
            f.write(f"  Avg distance after : {r['avg_dist_after']}\n")
            f.write(f"  Fooled model       : {r['fooled_pct']}%\n\n")

    print(f"\nResults saved to results/")
    print(f"    {json_path.name}")
    print(f"    {txt_path.name}\n")


if __name__ == "__main__":
    main()
