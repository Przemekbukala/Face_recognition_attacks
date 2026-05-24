"""
LFW dataset loader

After setup, <data_dir>/lfw-deepfunneled/ and <data_dir>/pairs.txt are normal files
and directories under your project
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

_OFFICIAL_PAIRS_TXT_URL = "https://ndownloader.figshare.com/files/5976006"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "lfw"

@dataclass
class LFWPair:
    img1_path: Path
    img2_path: Path
    is_same: bool
    name1: str
    name2: str

    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        return _load_rgb(self.img1_path), _load_rgb(self.img2_path)


@dataclass
class LFWPerson:
    name: str
    image_paths: List[Path]

    def load_image(self, idx: int = 0) -> np.ndarray:
        return _load_rgb(self.image_paths[idx])

    def load_all_images(self) -> List[np.ndarray]:
        return [_load_rgb(p) for p in self.image_paths]


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _img_path(images_dir: Path, name: str, num: int) -> Path:
    return images_dir / name / f"{name}_{num:04d}.jpg"


def _find_lfw_deepfunneled(root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for p in (root / "lfw-deepfunneled", root / "lfw-py" / "lfw-deepfunneled"):
        if p.is_dir():
            candidates.append(p)
    for p in root.rglob("lfw-deepfunneled"):
        if p.is_dir() and p not in candidates:
            candidates.append(p)
    for cand in candidates:
        try:
            for sub in cand.iterdir():
                if sub.is_dir() and any(sub.glob("*.jpg")):
                    return cand
        except OSError:
            continue
    return None


def _find_pairs_file(root: Path) -> Optional[Path]:
    """Locate LFW pair definition file inside a Kaggle archive tree."""
    direct = root / "lfw-py" / "pairs.txt"
    if direct.is_file():
        return direct
    found = [p for p in root.rglob("pairs.txt") if p.is_file()]
    if found:
        return min(found, key=lambda p: len(p.parts))
    for name in ("pairsDevTest.txt", "pairsDevTrain.txt"):
        for p in root.rglob(name):
            if p.is_file():
                return p
    return None


def _download_official_pairs_txt(dest: Path) -> None:
    """Download canonical pairs.txt (10×300 folds) when the image bundle omits it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"pairs.txt not in dataset archive, downloading official LFW list\n"
        f"  URL: {_OFFICIAL_PAIRS_TXT_URL}",
        flush=True,
    )
    req = Request(
        _OFFICIAL_PAIRS_TXT_URL,
        headers={"User-Agent": "Face_recognition_attacks/1.0"},
    )
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
    except URLError as e:
        raise RuntimeError(
            f"Failed to download pairs.txt from {_OFFICIAL_PAIRS_TXT_URL}: {e}"
        ) from e
    if not data or len(data) < 100:
        raise RuntimeError("Downloaded pairs.txt looks empty or truncated.")
    dest.write_bytes(data)
    print(f"  Saved: {dest}", flush=True)


def _copy_pairs_file(source: Path, dest: Path) -> None:
    """Copy pairs*.txt into data_dir"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.lstat(dest)
    except FileNotFoundError:
        pass
    else:
        dest.unlink()
    shutil.copy2(source.resolve(), dest)


def _materialize_image_tree(source: Path, dest: Path) -> None:
    """Copy source image tree to dest."""
    source = source.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        st = None
    if st is not None:
        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.copytree(source, dest)


class LFWDataset:
    """
    LFW deepfunneled images + pairs.txt.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        auto_download: bool = True,
        *,
        kaggle_dataset: str = "jessicali9530/lfw-dataset",
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.images_dir: Path = self.data_dir / "lfw-deepfunneled"
        self.pairs_file: Path = self.data_dir / "pairs.txt"
        self._kaggle_dataset = kaggle_dataset

        self._pairs: Optional[List[LFWPair]] = None
        self._people: Optional[List[LFWPerson]] = None

        if auto_download:
            self.ensure_data()

    def ensure_data(self) -> None:
        """Download LFW from Kaggle if needed and copy files under data_dir."""
        self._ensure_data()

    def download(self) -> None:
        """Alias for ensure_data, fetch dataset from Kaggle when missing locally."""
        self.ensure_data()

    def get_pairs(
        self,
        n: int = 500,
        only_same: bool = False,
        only_different: bool = False,
        seed: int = 42,
    ) -> List[LFWPair]:
        pairs = self._load_pairs()

        if only_same:
            pairs = [p for p in pairs if p.is_same]
        elif only_different:
            pairs = [p for p in pairs if not p.is_same]

        if n == -1 or n >= len(pairs):
            return pairs

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pairs), size=n, replace=False)
        return [pairs[i] for i in sorted(indices)]

    def get_people(
        self,
        n: int = 50,
        min_images: int = 2,
        seed: int = 42,
    ) -> List[LFWPerson]:
        people = self._load_people()
        filtered = [p for p in people if len(p.image_paths) >= min_images]

        if n == -1 or n >= len(filtered):
            return filtered

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(filtered), size=n, replace=False)
        return [filtered[i] for i in sorted(indices)]

    @property
    def n_images(self) -> int:
        if not self.images_dir.exists():
            return 0
        return sum(1 for _ in self.images_dir.rglob("*.jpg"))

    @property
    def n_people(self) -> int:
        return len(self._load_people())

    def _load_pairs(self) -> List[LFWPair]:
        if self._pairs is not None:
            return self._pairs

        if not self.pairs_file.exists():
            raise FileNotFoundError(
                f"pairs.txt not found at {self.pairs_file}. "
                "Use LFWDataset(auto_download=True) or place pairs.txt next to images."
            )

        pairs: List[LFWPair] = []
        with open(self.pairs_file, "r") as f:
            header = f.readline().strip().split()
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        if len(header) >= 2:
            n_folds, n_per_fold = int(header[0]), int(header[1])
            block = 2 * n_per_fold
            same_lines: List[str] = []
            diff_lines: List[str] = []
            for fold in range(n_folds):
                start = fold * block
                same_lines.extend(lines[start : start + n_per_fold])
                diff_lines.extend(
                    lines[start + n_per_fold : start + block]
                )
        else:
            n_each = int(header[0])
            same_lines = lines[:n_each]
            diff_lines = lines[n_each : n_each + n_each]

        for line in same_lines:
            parts = line.split("\t")
            name, n1, n2 = parts[0], int(parts[1]), int(parts[2])
            p1 = _img_path(self.images_dir, name, n1)
            p2 = _img_path(self.images_dir, name, n2)
            if p1.exists() and p2.exists():
                pairs.append(LFWPair(p1, p2, True, name, name))

        for line in diff_lines:
            parts = line.split("\t")
            name1, n1, name2, n2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1 = _img_path(self.images_dir, name1, n1)
            p2 = _img_path(self.images_dir, name2, n2)
            if p1.exists() and p2.exists():
                pairs.append(LFWPair(p1, p2, False, name1, name2))

        self._pairs = pairs
        return pairs

    def _load_people(self) -> List[LFWPerson]:
        if self._people is not None:
            return self._people

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}. "
                "Use LFWDataset(auto_download=True)."
            )

        people: List[LFWPerson] = []
        for person_dir in sorted(self.images_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            imgs = sorted(person_dir.glob("*.jpg"))
            if imgs:
                people.append(LFWPerson(name=person_dir.name, image_paths=imgs))

        self._people = people
        return people

    def _have_local_bundle(self) -> bool:
        img = self.data_dir / "lfw-deepfunneled"
        pr = self.data_dir / "pairs.txt"
        if not pr.is_file() or not img.is_dir():
            return False
        return any(img.iterdir())

    def _ensure_data(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        img_dir = self.data_dir / "lfw-deepfunneled"
        pairs_path = self.data_dir / "pairs.txt"

        if self._have_local_bundle():
            self.images_dir = self.data_dir / "lfw-deepfunneled"
            self.pairs_file = self.data_dir / "pairs.txt"
            return

        if img_dir.is_dir() and any(img_dir.iterdir()) and not pairs_path.is_file():
            print(
                "Images found under data/lfw but pairs.txt is missing — "
                "downloading official pairs.txt …",
                flush=True,
            )
            _download_official_pairs_txt(pairs_path)
            self.images_dir = img_dir
            self.pairs_file = pairs_path
            print(f"LFW ready at {self.data_dir}", flush=True)
            return

        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "LFWDataset needs kagglehub. Install: pip install kagglehub"
            ) from e

        print(f"Downloading LFW via kagglehub ({self._kaggle_dataset}) …", flush=True)
        root = Path(kagglehub.dataset_download(self._kaggle_dataset))
        print(f"Path to dataset files: {root}")

        images = _find_lfw_deepfunneled(root)
        if images is None:
            raise RuntimeError(
                f"Could not find lfw-deepfunneled with images under {root}"
            )
        pairs = _find_pairs_file(root)
        local_images = self.data_dir / "lfw-deepfunneled"
        local_pairs = self.data_dir / "pairs.txt"
        print("Copying LFW images into data_dir …", flush=True)
        _materialize_image_tree(images, local_images)
        self.images_dir = local_images

        if pairs is not None:
            _copy_pairs_file(pairs, local_pairs)
            self.pairs_file = local_pairs
        else:
            print(
                "No pairs*.txt in Kaggle archive (common for newer versions).",
                flush=True,
            )
            _download_official_pairs_txt(local_pairs)
            self.pairs_file = local_pairs
        print(f"LFW ready at {self.data_dir}", flush=True)


if __name__ == "__main__":
    ds = LFWDataset()
    print(f"People : {ds.n_people}")
    print(f"Images : {ds.n_images}")

    pairs = ds.get_pairs(n=10)
    print("\nFirst 5 pairs:")
    for p in pairs[:5]:
        label = "SAME" if p.is_same else "DIFF"
        print(f"  [{label}] {p.name1} vs {p.name2}")

    people = ds.get_people(n=5, min_images=3)
    print("\nSample people (≥3 images):")
    for person in people:
        print(f"  {person.name}: {len(person.image_paths)} images")
