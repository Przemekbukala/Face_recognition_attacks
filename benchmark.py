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
from src.attacks.feature_level.feature_attacks import ffm_attack, fim_attack
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

    "ffm": lambda img, **p: ffm_attack(img, **p),

    "fim": lambda img, **p: fim_attack(img, **p),

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

    ATTACKS.append((name, fn, params))

#  Benchmark 

def run_benchmark(pairs, threshold: float) -> dict:
    all_results = {}

    for idx, (attack_name, attack_fn, attack_params) in enumerate(ATTACKS, start=1):
        # create a unique key for the attack so multiple attacks with the same
        # base name but different parameters don't overwrite each other
        attack_key = f"{attack_name}#{idx}"
        attack_display = f"{attack_name} {json.dumps(attack_params, ensure_ascii=False)}"

        print(f"\n{'='*60}")
        print(f"  Attack: {attack_display}  |  pairs: {len(pairs)}  |  threshold: {threshold}")
        print(f"{'='*60}")

        rows = []
        n_skipped = 0
        n_skipped_after = 0
        t0 = time.time()

        n_before_same = 0
        n_before_different = 0

        for pair in tqdm(pairs, desc=attack_key, unit="pair"):
            img1, img2 = pair.load_images()

            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)

            dist_before, label_before = compare_embeddings(
                emb1, emb2, threshold
            )

            if label_before == "Invalid embeddings":
                n_skipped += 1
                continue

            if label_before == "Same person":
                n_before_same += 1
            elif label_before == "Different people":
                n_before_different += 1

            img1_attacked = attack_fn(img1, **attack_params)

            emb1_attacked = get_embedding(img1_attacked)
            emb2_attacked = emb2  # unchanged

            dist_after, label_after = compare_embeddings(
                emb1_attacked, emb2_attacked, threshold
            )

            if label_after == "Invalid embeddings":
                n_skipped_after += 1
                continue

            fooled = label_after != label_before

            tqdm.write(
                f"  {'[FOOLED]  ' if fooled else '[NO CHANGE]'}  "
                f"{pair.img1_path.stem} vs {pair.img2_path.stem}  "
                f"{dist_before:.4f} -> {dist_after:.4f}  ({label_after})"
            )

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

        fooled_pct = (
            round(sum(1 for r in rows if r["fooled"]) / evaluated * 100, 1)
            if evaluated else float("nan")
        )

        print(f"\n{'─'*60}")
        print(f"  Evaluated: {evaluated}   Skipped before attack: {n_skipped}   Skipped after attack: {n_skipped_after}   Time: {elapsed:.1f}s")
        print(f"  Avg distance before attack : {avg(rows, 'distance_before'):.4f}")
        print(f"  Avg distance after attack  : {avg(rows, 'distance_after'):.4f}")

        print(f"\n  Before attack:")
        print(f"    Same person      : {n_before_same}")
        print(f"    Different people : {n_before_different}")

        print(f"  Attack fooled model : {fooled_pct:.1f}%")

        all_results[attack_key] = {
            "n_evaluated": evaluated,
            "n_skipped": n_skipped,
            "n_skipped_after": n_skipped_after,
            "elapsed_seconds": round(elapsed, 1),
            "threshold": threshold,
            "avg_dist_before": avg(rows, "distance_before"),
            "avg_dist_after": avg(rows, "distance_after"),

            "fooled_pct": fooled_pct,

            "before_same": n_before_same,
            "before_different": n_before_different,

            "parameters": attack_params,
            "display_name": attack_display,
            "pairs": rows,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=-1,
                        help="Number of pairs (-1 = all)")
    parser.add_argument("--threshold", type=float, default=0.7233)
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
    # build a readable attacks description including parameters
    attacks_desc = [f"{name} {params}" for name, _, params in ATTACKS]
    print(f"Starting benchmark: {len(pairs)} pairs, attacks: {attacks_desc}\n")
    results = run_benchmark(pairs, threshold=args.threshold) 

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
            params_str = json.dumps(r.get('parameters', {}), ensure_ascii=False)
            f.write(f"Attack: {attack_name}  Parameters: {params_str}\n")
            f.write(f"  Evaluated: {r['n_evaluated']}  Skipped before attack: {r['n_skipped']}  Skipped after attack: {r['n_skipped_after']}  Time: {r['elapsed_seconds']}s\n")
            f.write(f"  Avg distance before: {r['avg_dist_before']}\n")
            f.write(f"  Avg distance after : {r['avg_dist_after']}\n")
            f.write(f"  Fooled model       : {r['fooled_pct']}%\n\n")

    all_txt_path = RESULTS_DIR / "all_results.txt"
    with open(all_txt_path, "a", encoding="utf-8") as f:
        f.write(f"Benchmark  {ts}\n")
        f.write(f"Pairs: {len(pairs)}  Threshold: {args.threshold}\n\n")
        for attack_name, r in results.items():
            params_str = json.dumps(r.get('parameters', {}), ensure_ascii=False)
            f.write(f"Attack: {attack_name}  Parameters: {params_str}\n")
            f.write(f"  Evaluated: {r['n_evaluated']}  Skipped before attack: {r['n_skipped']}  Skipped after attack: {r['n_skipped_after']}  Time: {r['elapsed_seconds']}s\n")
            f.write(f"  Avg distance before: {r['avg_dist_before']}\n")
            f.write(f"  Avg distance after : {r['avg_dist_after']}\n")
            f.write(f"  Fooled model       : {r['fooled_pct']}%\n\n")

    print(f"\nResults saved to results/")
    print(f"    {json_path.name}")
    print(f"    {txt_path.name}\n")


if __name__ == "__main__":
    main()
