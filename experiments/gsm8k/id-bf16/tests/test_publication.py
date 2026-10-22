"""Publication checks exercise corruption detection and portable configuration."""
from pathlib import Path

import pytest

from gsm8k_id.common import configuration, runtime_configuration, write
from gsm8k_id.results import check_hashes, verify_results


def test_paths_resolve_from_workspace_not_process_directory(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    models = tmp_path / "shared-models"
    write(root / "experiment.json", {"models": [{"path": "models/base"}, {"path": str(models)}]})
    write(root / "runtime.json", {"historical_root": "historical-cot", "gpu_lock": ".gpu0.lock"})
    monkeypatch.chdir(tmp_path)
    actual = configuration(root)
    assert actual["models"][0]["path"] == str(root / "models/base")
    assert actual["models"][1]["path"] == str(models)
    assert runtime_configuration(root)["gpu_lock"] == str(root / ".gpu0.lock")


def test_manifest_rejects_corrupt_file_and_path_escape(tmp_path):
    from gsm8k_id.common import sha
    payload = tmp_path / "result.csv"
    payload.write_text("original")
    original = sha(payload)
    check_hashes(tmp_path, {"result.csv": original})
    payload.write_text("changed")
    with pytest.raises(ValueError, match="Published file changed"):
        check_hashes(tmp_path, {"result.csv": original})
    with pytest.raises(ValueError, match="Invalid manifest path"):
        check_hashes(tmp_path, {"../outside": original})


def test_retained_results_are_independent_of_model_and_raw_directories():
    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/statistics.json").exists():
        pytest.skip("Fresh inference workspace has no completed published results")
    result = verify_results(root)
    assert result["question_records"] == 5276
    assert result["common_complete_questions"] == 1299
    assert not result["raw_answers_rescored"]
    assert not result["activation_geometry_recomputed"]


def test_bootstrap_uses_sibling_code_and_validates_prepared_bytes(tmp_path):
    import sys
    from gsm8k_id.bootstrap import bootstrap
    from gsm8k_id.common import sha
    root, protocol = tmp_path / "run", tmp_path / "cot"
    root.mkdir()
    protocol.mkdir()
    write(root / "experiment.json", {"models": [{"stage": "base", "path": "models/base"}]})
    write(root / "runtime.json", {"historical_root": "history", "gpu_lock": ".lock"})
    write(root / "reference/experiment.json", {"models": [{"stage": "base", "path": "unused"}]})
    expected = tmp_path / "expected"
    expected.write_text("fixed input\n")
    write(root / "reference/data.json", {"prepared_files": {"data/input.txt": sha(expected)}, "source_files": {}})
    write(root / "reference/tokenization.json", {"files": {"data/tokens.txt": sha(expected)}})
    (protocol / "common.py").write_text("ROOT = None\n")
    (protocol / "prepare.py").write_text("from common import ROOT\n(ROOT / 'data').mkdir(exist_ok=True)\n(ROOT / 'data/input.txt').write_text('fixed input\\n')\n")
    (protocol / "tokenize_panel.py").write_text("from common import ROOT\n(ROOT / 'data/tokens.txt').write_text('fixed input\\n')\n")
    previous = sys.modules.get("common")
    bootstrap(root, protocol)
    assert sys.modules.get("common") is previous
    (protocol / "tokenize_panel.py").write_text("from common import ROOT\n(ROOT / 'data/tokens.txt').write_text('changed tokenization')\n")
    with pytest.raises(ValueError, match="Reconstructed tokens differ"):
        bootstrap(root, protocol)


def test_report_cli_requires_separate_output_and_does_not_modify_input(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-m", "gsm8k_id.report"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "--output-dir" in result.stderr
