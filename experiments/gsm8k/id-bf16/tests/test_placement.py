from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from gsm8k_id import engine


class Model:
    def __init__(self, device="cuda:0", dtype=torch.bfloat16):
        self.parameter = SimpleNamespace(dtype=dtype, device=torch.device(device))
        self.config = SimpleNamespace(num_hidden_layers=32, hidden_size=4096)

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def parameters(self):
        return iter([self.parameter])


def setup(monkeypatch, model):
    monkeypatch.setattr(engine, "configure", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    return calls


def test_load_uses_bf16_and_a_single_explicit_gpu(monkeypatch):
    model = Model()
    calls = setup(monkeypatch, model)
    assert engine.load_model("local", {"seed": 42, "dtype": "bfloat16", "device": "cuda:0"}) is model
    assert calls[0]["torch_dtype"] == torch.bfloat16
    assert calls[0]["device_map"] == {"": 0}
    assert "max_memory" not in calls[0]


@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda:0", torch.float32)])
def test_load_rejects_cpu_offload_or_fp32_weights(monkeypatch, device, dtype):
    setup(monkeypatch, Model(device, dtype))
    with pytest.raises(RuntimeError, match="All BF16 parameters"):
        engine.load_model("local", {"seed": 42, "dtype": "bfloat16", "device": "cuda:0"})


def test_load_rejects_device_without_bf16_support(monkeypatch):
    calls = setup(monkeypatch, Model())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="Native BF16"):
        engine.load_model("local", {"seed": 42, "dtype": "bfloat16", "device": "cuda:0"})
    assert calls == []
