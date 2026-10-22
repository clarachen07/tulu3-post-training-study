import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from gsm8k_id.engine import Capture, configure, extract, greedy


def tiny():
    configure(42)
    cfg = transformers.LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                    max_position_embeddings=128, eos_token_id=99, attention_dropout=0)
    cfg._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(cfg).to(dtype=torch.bfloat16).eval()


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(65 + i) for i in ids)


def test_cached_greedy_equals_full_prefix_argmax_and_is_repeatable():
    m = tiny()
    ids = [1, 5, 7, 9]
    config = {"max_new_tokens": 9, "stop": "unmatchable"}
    actual = greedy(m, Tokenizer(), ids, config)
    again = greedy(m, Tokenizer(), ids, config)
    assert actual["output_token_ids"] == again["output_token_ids"]
    expected = []
    with torch.inference_mode():
        for _ in range(9):
            inp = torch.tensor([ids + expected])
            out = m(input_ids=inp, use_cache=False)
            expected.append(int(out.logits[0, -1].argmax()))
    assert actual["output_token_ids"] == expected
    assert actual["truncated"] and actual["finish_reason"] == "length"


def test_positions_match_raw_layers_and_final_token_not_predicting_token():
    m = tiny()
    prompt, output = [1, 5, 7, 9], [4, 3, 2, 1, 6]
    actual, anchors, pos, error = extract(m, prompt, output, check_norm=True)
    capture = Capture(m.model, list(range(len(prompt) + len(output))))
    with torch.inference_mode():
        m.model(input_ids=torch.tensor([prompt + output]), use_cache=False)
    capture.close()
    expected = torch.stack([capture.values[i] for i in range(3)], dim=1).numpy()
    np.testing.assert_array_equal(actual, expected[pos])
    assert pos[-1] == len(prompt) + len(output) - 1
    assert error <= 2 * torch.finfo(torch.bfloat16).eps and anchors == list(range(5))
    assert actual.dtype == np.float32
    # Every saved float32 is exactly a BF16 value, with no invented extra precision.
    np.testing.assert_array_equal(torch.from_numpy(actual).bfloat16().float().numpy(), actual)
    # Last raw block differs from post-RMSNorm output; hooks have all been removed.
    assert all(not layer._forward_hooks for layer in m.model.layers)


def test_cached_and_uncached_hidden_states_agree():
    m = tiny()
    all_ids = [1, 5, 7, 9, 4, 3]
    with torch.inference_mode():
        full = m.model(input_ids=torch.tensor([all_ids]), use_cache=False).last_hidden_state
        prefix = m.model(input_ids=torch.tensor([all_ids[:-1]]), use_cache=True)
        last = m.model(input_ids=torch.tensor([[all_ids[-1]]]), past_key_values=prefix.past_key_values,
                       attention_mask=torch.ones(1, len(all_ids), dtype=torch.long), use_cache=True)
    torch.testing.assert_close(last.last_hidden_state[0, -1], full[0, -1], rtol=2 * torch.finfo(torch.bfloat16).eps, atol=1e-6)


def test_empty_output_has_only_input_anchor():
    m = tiny()
    x, anchors, pos, _ = extract(m, [1, 2, 3], [], check_norm=True)
    assert x.shape == (1, 3, 16) and anchors == [0] and pos == [2, None, None, None, None]


def test_first_token_eos_is_not_geometry_token():
    m = tiny()
    prompt = [1, 2, 3]
    first = greedy(m, Tokenizer(), prompt, {"max_new_tokens": 1, "stop": "unmatchable"})["output_token_ids"][0]
    m.config.eos_token_id = first
    out = greedy(m, Tokenizer(), prompt, {"max_new_tokens": 9, "stop": "unmatchable"})
    assert out["finish_reason"] == "eos" and out["effective_token_ids"] == [] and out["continuation"] == ""


def test_fp32_forward_is_rejected_before_data_can_be_saved():
    m = tiny().float()
    with pytest.raises(RuntimeError, match="non-BF16"):
        greedy(m, Tokenizer(), [1, 2, 3], {"max_new_tokens": 2, "stop": "unmatchable"})
    with pytest.raises(RuntimeError, match="Non-BF16"):
        extract(m, [1, 2, 3], [4], check_norm=True)
    assert all(not layer._forward_hooks for layer in m.model.layers)
