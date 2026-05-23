"""
LFW attack evaluator using people_comparer.compare_embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from src.lfw_dataset import LFWDataset, LFWPair

_comparer = None


def _get_comparer():
    """comparer so importing this module does not pull heavy deps."""
    global _comparer
    if _comparer is None:
        from face_recognition_model.people_comparer import (
            get_embedding,
            compare_embeddings,
        )
        _comparer = (get_embedding, compare_embeddings)
    return _comparer


@dataclass
class PairVerdict:
    """One pair: outputs of compare_embeddings before / after attack on img1."""

    distance_before: float
    label_before: str
    distance_after: float
    label_after: str


@dataclass
class AttackResult:
    """Roll-up of one attack run, split by ground-truth same vs. different identity
    """

    attack_name: str
    n_pairs_evaluated: int
    threshold: float
    same_pairs: List[PairVerdict] = field(default_factory=list)
    diff_pairs: List[PairVerdict] = field(default_factory=list)
    n_skipped: int = 0
    elapsed_seconds: float = 0.0

    def _mean_dist(self, rows: List[PairVerdict], before: bool) -> float:
        if not rows:
            return float("nan")
        key = "distance_before" if before else "distance_after"
        return float(np.mean([getattr(r, key) for r in rows]))

    def summary(self) -> str:
        """Summary using only distance + labels."""
        s_same = self.same_pairs
        s_diff = self.diff_pairs

        def frac(labels: List[str], target: str) -> float:
            if not labels:
                return float("nan")
            return sum(1 for x in labels if x == target) / len(labels)

        same_after_diff = frac([r.label_after for r in s_same], "Different person")
        diff_after_same = frac([r.label_after for r in s_diff], "Same person")

        lines = [
            f"{'='*60}",
            f"  Attack : {self.attack_name}",
            f"  Pairs  : {self.n_pairs_evaluated}  |  threshold = {self.threshold:.2f}",
            f"  Skipped: {self.n_skipped}  (no valid compare_embeddings before attack)",
            f"  Time   : {self.elapsed_seconds:.1f} s",
            f"{'─'*60}",
            "  (All numbers from compare_embeddings: distance + Same/Different/Invalid)",
            f"  Same-person pairs   — avg distance before: {self._mean_dist(s_same, True):.4f}",
            f"  Same-person pairs   — avg distance after : {self._mean_dist(s_same, False):.4f}",
            f"  Same-person pairs   — after: \"Different person\" rate: {same_after_diff*100:.1f}%",
            f"  Diff-person pairs   — avg distance before: {self._mean_dist(s_diff, True):.4f}",
            f"  Diff-person pairs   — avg distance after : {self._mean_dist(s_diff, False):.4f}",
            f"  Diff-person pairs   — after: \"Same person\" rate (false accept): {diff_after_same*100:.1f}%",
            f"{'='*60}",
        ]
        return "\n".join(lines)


AttackFn = Callable[[np.ndarray], np.ndarray]


def evaluate_attack(
    attack_fn: AttackFn,
    attack_name: str,
    pairs: Optional[List[LFWPair]] = None,
    n_pairs: int = 300,
    threshold: float = 0.45,
    seed: int = 42,
    verbose: bool = True,
) -> AttackResult:
    """Run attack_fn on img1 for each pair, count skips on invalid baseline or attack errors.
    """
    if pairs is None:
        ds = LFWDataset()
        n_same = n_pairs // 2
        n_diff = n_pairs - n_same
        pairs = ds.get_pairs(n=n_same, only_same=True, seed=seed) + ds.get_pairs(
            n=n_diff, only_different=True, seed=seed
        )

    get_embedding, compare_embeddings = _get_comparer()

    out = AttackResult(
        attack_name=attack_name,
        n_pairs_evaluated=len(pairs),
        threshold=threshold,
    )

    t0 = time.time()
    iterator = tqdm(pairs, desc=attack_name, unit="pair") if verbose else pairs

    for pair in iterator:
        img1, img2 = pair.load_images()
        emb1_clean = get_embedding(img1)
        emb2 = get_embedding(img2)

        dist_before, label_before = compare_embeddings(emb1_clean, emb2, threshold)
        if label_before == "Invalid embeddings":
            out.n_skipped += 1
            continue

        try:
            img1_attacked = attack_fn(img1)
        except Exception:
            out.n_skipped += 1
            continue

        emb1_attacked = get_embedding(img1_attacked)
        dist_after, label_after = compare_embeddings(emb1_attacked, emb2, threshold)

        row = PairVerdict(dist_before, label_before, dist_after, label_after)
        if pair.is_same:
            out.same_pairs.append(row)
        else:
            out.diff_pairs.append(row)

    out.elapsed_seconds = time.time() - t0
    return out
