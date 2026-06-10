import logging
import sys
from pathlib import Path
from unittest import result
import numpy as np
import csv
from PIL import Image
from utils.params_loader import get_attack_params
from face_recognition_model.people_comparer import get_embedding, compare_embeddings
from src.lfw_dataset import LFWDataset, LFWPerson
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_RESULTS_DIR = _REPO_ROOT / "results" / "bit_flip"
writer = SummaryWriter(log_dir=str(_REPO_ROOT / "results" / "bit_flip" / "tensorboard_logs"))
def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _save_image(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img).save(path)


def _show_image(img: np.ndarray, title: str) -> None:
    Image.fromarray(img).show(title=title)

def _evaluate_similarity(photo1: np.ndarray, emb2: np.ndarray) -> float:
    """Evaluate the similarity between two photos using the face recognition model."""
    emb1 = get_embedding(photo1)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    cosine_distance = 1 - np.dot(emb1, emb2)
    return cosine_distance

def _set_up_training(size_x : int, size_y : int) -> np.array:
    """Set up the training data for the genetic algorithm."""
    population = np.zeros((get_attack_params("bitflip").get("population_size"), get_attack_params("bitflip").get("bits_to_flip") * 2), dtype=np.uint8)
    for genome in population:
        for i in range(0, len(genome), 2):
            genome[i] = np.random.randint(0, size_x)
            genome[i + 1] = np.random.randint(0, size_y)
    logger.debug("Population initialized")
    return population

def bit_flip_based_on_genome(photo: np.ndarray, genome: np.array) -> np.ndarray:
    """Apply the bit flip attack to the photo based on the genome."""
    attacked_photo = photo.copy()
    for i in range(0, len(genome), 2):
        x = genome[i]
        y = genome[i + 1]
        attacked_photo[x, y] = np.bitwise_xor(attacked_photo[x, y], 0x80)  # Flip the pixel value
    return attacked_photo

def _mutate(genome: np.array, size_x : int, size_y : int) -> np.array:
    """Mutate the genome by randomly changing some of its values."""
    mutation_rate = get_attack_params("bitflip").get("mutation_chance")
    for i in range(0, len(genome), 2):
        if np.random.rand() < mutation_rate:
            genome[i] = np.random.randint(0, size_x)
            genome[i + 1] = np.random.randint(0, size_y)
    return genome

def _create_new_genome(genome1: np.array, genome2: np.array) -> np.array:
    index = np.random.randint(0, len(genome1))
    return np.concatenate((genome1[:index], genome2[index:]))

def _choose_population_to_futher_training(population : np.array, photo_to_attack: np.ndarray, photos: list) -> np.array:
    """Choose a subset of the population for further training."""
    
    results_of_similarity = np.array([np.sum([_evaluate_similarity(bit_flip_based_on_genome(photo_to_attack, genome), second_photo) for second_photo in photos]) for genome in population])
    indexes = np.argsort(-results_of_similarity)
    population = population[indexes]
    index = int(get_attack_params("bitflip").get("percent_of_population_to_keep") * len(population))
    return population[:index]

def _train_attack(photos : list) -> np.ndarray:
    """Train the bit flip attack using a genetic algorithm."""
    photo1 = photos[0]
    population = _set_up_training(photo1.shape[0], photo1.shape[1])
    best_fitted_genome = population[0]
    photos_embeddings = np.array([get_embedding(photo) for photo in photos])
    for i in range(get_attack_params("bitflip").get("epochs")):
        population = _choose_population_to_futher_training(population, photo1, photos_embeddings)
        new_population = []
        for __ in range(get_attack_params("bitflip").get("population_size")):
            genome1 = population[np.random.randint(0, len(population))]
            genome2 = population[np.random.randint(0, len(population))]
            new_genome = _create_new_genome(genome1, genome2)
            new_genome = _mutate(new_genome, photo1.shape[0], photo1.shape[1])
            new_population.append(new_genome)
        best_fitted_genome = max(population[0], best_fitted_genome, key=lambda genome: np.sum([_evaluate_similarity(bit_flip_based_on_genome(photo1, genome), second_photo) for second_photo in photos_embeddings]))
        best_similarity = np.sum([_evaluate_similarity(bit_flip_based_on_genome(photo1, best_fitted_genome), second_photo) for second_photo in photos_embeddings])
        logger.debug(f"Best similarity: {best_similarity} in epoch {i}")
        writer.add_scalar("Simiratiy/train", best_similarity, i)
        population = np.array(new_population)
    writer.close()
    return best_fitted_genome

if __name__ == "__main__":
    photo_set : LFWDataset = LFWDataset()
    logger.debug("Reading data")
    photo_set.ensure_data()
    logger.debug("Data readed")
    person : LFWPerson = photo_set.get_people()[0]
    photos : list = person.load_all_images()
    results_dir = _ensure_results_dir()
    _show_image(photos[0], "before_training_0")
    _save_image(photos[0], results_dir / "before_training_0.png")
    ref_emb = get_embedding(photos[0])
    before_results = []
    for photo in photos[1:]:
        before_results.append(compare_embeddings(ref_emb, get_embedding(photo)))
    logger.info(before_results)
    best_attack = _train_attack(photos)
    attacked = bit_flip_based_on_genome(photos[0], best_attack)
    _show_image(attacked, "after_training_0")
    _save_image(attacked, results_dir / "after_training_0.png")
    attacked_emb = get_embedding(attacked)
    after_results = []
    for photo in photos[1:]:
        after_results.append(compare_embeddings(attacked_emb, get_embedding(photo)))
    logger.info(after_results)
    eval_path = results_dir / "evaluation.csv"
    with eval_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["photo_index", "before_distance", "before_result", "after_distance", "after_result"])
        for idx, (before, after) in enumerate(zip(before_results, after_results, strict=True), start=1):
            writer.writerow([idx, before[0], before[1], after[0], after[1]])
