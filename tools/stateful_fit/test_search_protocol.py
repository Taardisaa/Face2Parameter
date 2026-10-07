"""Known-invalid stateless ABMX must not enter candidate optimization."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/base_comparison"))
from search import optimize


class SearchProtocolTests(unittest.TestCase):
    def test_stateless_abmx_is_rejected_before_candidate_work(self):
        model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "explicit stateful"):
            optimize(None, model, None, None, {}, "native+ABMX", 1)


if __name__ == "__main__":
    unittest.main()
