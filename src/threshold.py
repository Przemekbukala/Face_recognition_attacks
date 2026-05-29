from src.lfw_dataset import LFWDataset
from face_recognition_model.people_comparer import (
    get_embedding,
    compare_embeddings
)

import numpy as np


def compute_distances(pairs):

    distances = []

    for idx, pair in enumerate(pairs):

        print(f"Processing pair {idx+1}/{len(pairs)}")

        img1, img2 = pair.load_images()

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        dist, _ = compare_embeddings(
            emb1,
            emb2,
            threshold=0.5
        )

        distances.append(dist)
        print(f"Distance: {dist:.4f}")
    return distances


def determine_threshold(
    dataset,
    n_same=200,
    n_different=200,
    percentile=95
):

    same_pairs = dataset.get_pairs(
        n=n_same,
        only_same=True
    )

    different_pairs = dataset.get_pairs(
        n=n_different,
        only_different=True
    )

    print("\nComputing SAME-person distances...")
    distances_same = compute_distances(same_pairs)

    print("\nComputing DIFFERENT-person distances...")
    distances_different = compute_distances(different_pairs)

    threshold = (np.percentile(distances_same, 95) + np.percentile(distances_different, 5)) / 2

    print("\n==============================")
    print(f"Same pairs: {len(distances_same)}")
    print(f"Different pairs: {len(distances_different)}")
    print(f"Threshold ({percentile} percentile): {threshold:.4f}")

    return threshold


if __name__ == "__main__":

    dataset = LFWDataset("data/lfw")

    threshold = determine_threshold(
        dataset,
        n_same=200,
        n_different=200,
        percentile=95
    )

    print(f"\nRecommended threshold: {threshold:.4f}")