import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ewt_eval import pipeline


class PipelineTests(unittest.TestCase):
    def test_unexpected_failure_does_not_blindly_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "logs").mkdir()
            log = root / "logs/error.log"
            log.write_text("INFO rest of memory reserved for KV Cache\nRuntimeError: malformed internal request\n")
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "config", return_value={}), patch.object(pipeline, "state"), patch.object(pipeline, "worker", return_value=(1, log)) as worker:
                with self.assertRaises(RuntimeError):
                    pipeline.preflight()
                self.assertEqual(worker.call_count, 1)

    def test_memory_sweep_and_compatibility_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            log = root / "error.log"
            log.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory")
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "config", return_value={}), patch.object(pipeline, "state"), patch.object(pipeline, "worker", return_value=(1, log)) as worker:
                with self.assertRaises(RuntimeError):
                    pipeline.preflight()
                self.assertEqual(worker.call_count, 4)
                self.assertEqual([c.args[0][c.args[0].index("--max-num-seqs") + 1] for c in worker.call_args_list], ["4", "2", "1", "1"])
            log.write_text("ImportError: undefined symbol")
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "config", return_value={}), patch.object(pipeline, "state"), patch.object(pipeline, "worker", return_value=(1, log)) as worker:
                with self.assertRaises(RuntimeError):
                    pipeline.preflight()
                self.assertEqual(worker.call_count, 2)
                self.assertEqual(worker.call_args_list[-1].args[0][-1], "transformers")


if __name__ == "__main__":
    unittest.main()
