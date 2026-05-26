import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from utils.params_loader import get_attack_params


class TestGetAttackParams(unittest.TestCase):
	def test_returns_bitflip_params(self) -> None:
		params = get_attack_params("bitflip")
		self.assertEqual(params.get("bits_to_flip"), 50)

if __name__ == "__main__":
	unittest.main()