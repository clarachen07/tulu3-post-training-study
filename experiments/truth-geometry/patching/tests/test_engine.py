"""Independent causal and hook-isolation checks on a tiny, local Llama."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from truth_patching.engine import capture, extract_hidden, forward, replace_hidden


@pytest.fixture(scope="module")
def llama():
    torch.manual_seed(20261007)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).cpu().float().eval()


@pytest.fixture
def inputs():
    recipient = torch.tensor([[1, 9, 12, 18, 25, 31, 40]])
    donor = recipient.clone()
    donor[0, 3] = 19
    return recipient, donor


@pytest.mark.parametrize("as_tuple", [False, True])
def test_replace_preserves_structure_and_changes_only_requested_vector(as_tuple):
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    original = hidden.clone()
    metadata = object()
    output = (hidden, metadata) if as_tuple else hidden
    donor_vector = torch.tensor([-11.0, -12.0, -13.0, -14.0])

    assert extract_hidden(output) is hidden
    replaced = replace_hidden(output, 2, donor_vector)
    actual = extract_hidden(replaced)
    expected = original.clone()
    expected[0, 2] = donor_vector

    assert isinstance(replaced, tuple) == as_tuple
    if as_tuple:
        assert len(replaced) == len(output)
        assert replaced[1] is metadata
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    assert actual.data_ptr() != hidden.data_ptr()


def test_capture_and_observation_preserve_model_predictions(llama, inputs):
    recipient, _ = inputs
    with torch.no_grad():
        reference = llama(recipient, use_cache=False).logits.float().cpu()
    plain = forward(llama, recipient)
    observed = forward(llama, recipient, observe=True)
    captured, activations = capture(llama, recipient)

    for logits in (plain, observed, captured):
        assert logits.shape == (1, recipient.shape[1], llama.config.vocab_size)
        assert logits.device.type == "cpu"
        assert logits.dtype == torch.float32
        assert not logits.requires_grad
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-6)
    assert set(activations) == set(range(llama.config.num_hidden_layers))
    for hidden in activations.values():
        assert hidden.shape == (1, recipient.shape[1], llama.config.hidden_size)
        assert hidden.device.type == "cpu"
        assert hidden.dtype == next(llama.parameters()).dtype
        assert not hidden.requires_grad


def test_self_patch_and_subsequent_plain_forward_are_unchanged(llama, inputs):
    recipient, _ = inputs
    reference, activations = capture(llama, recipient)
    patched = forward(llama, recipient, patch=(1, 4, activations[1][0, 4]))
    subsequent = forward(llama, recipient)
    torch.testing.assert_close(patched, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(subsequent, reference, rtol=1e-5, atol=1e-6)


def test_patch_before_first_changed_token_has_no_effect(llama, inputs):
    recipient, donor = inputs
    reference = forward(llama, recipient)
    _, donor_activations = capture(llama, donor)
    for layer in range(llama.config.num_hidden_layers):
        patched = forward(
            llama, recipient, patch=(layer, 2, donor_activations[layer][0, 2])
        )
        torch.testing.assert_close(patched, reference, rtol=1e-5, atol=1e-6)


def test_last_layer_nonfinal_patch_cannot_change_final_logits(llama, inputs):
    recipient, donor = inputs
    reference = forward(llama, recipient)
    _, donor_activations = capture(llama, donor)
    last_layer = llama.config.num_hidden_layers - 1
    patched = forward(
        llama, recipient, patch=(last_layer, 3, donor_activations[last_layer][0, 3])
    )
    torch.testing.assert_close(patched[:, -1], reference[:, -1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(patched[:, 3], reference[:, 3], rtol=1e-5, atol=1e-6)


def test_last_layer_final_patch_recovers_donor_logits_and_does_not_leak(llama, inputs):
    recipient, donor = inputs
    recipient_logits = forward(llama, recipient)
    donor_logits, donor_activations = capture(llama, donor)
    last_layer = llama.config.num_hidden_layers - 1
    last_position = recipient.shape[1] - 1
    patched = forward(
        llama,
        recipient,
        patch=(
            last_layer,
            last_position,
            donor_activations[last_layer][0, last_position],
        ),
    )
    torch.testing.assert_close(
        patched[:, -1], donor_logits[:, -1], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        patched[:, :-1], recipient_logits[:, :-1], rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(
        donor_logits[:, -1], recipient_logits[:, -1], rtol=1e-5, atol=1e-6
    )
    after_patch = forward(llama, recipient)
    torch.testing.assert_close(after_patch, recipient_logits, rtol=1e-5, atol=1e-6)


def test_failed_patch_removes_temporary_hooks(llama, inputs):
    recipient, _ = inputs
    reference = forward(llama, recipient)
    invalid_vector = torch.zeros(llama.config.hidden_size + 1)
    with pytest.raises(ValueError):
        forward(llama, recipient, patch=(1, 3, invalid_vector), observe=True)
    after_error = forward(llama, recipient)
    torch.testing.assert_close(after_error, reference, rtol=1e-5, atol=1e-6)
