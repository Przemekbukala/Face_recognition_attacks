import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from utils.params_loader import get_attack_params

print(get_attack_params("bitflip"))