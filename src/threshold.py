from src.lfw_dataset import LFWDataset
from face_recognition_model.people_comparer import get_embedding, compare_embeddings

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