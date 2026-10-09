import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tulu_id.extract import commit_batch, verify_batch, LastTokenCapture


class BatchTests(unittest.TestCase):
    def test_atomic_batch_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'batch_000000.npz'
            values = np.ones((2, 2, 8), np.float32)
            commit_batch(p, ['a', 'b'], values, 'identity')
            self.assertTrue(verify_batch(p, ['a', 'b'], 'identity', 2, 8))
            with self.assertRaises(RuntimeError): verify_batch(p, ['b', 'a'], 'identity', 2, 8)
            with self.assertRaises(RuntimeError): verify_batch(p, ['a', 'b'], 'other', 2, 8)
            with p.open('ab') as f: f.write(b'corruption')
            with self.assertRaises(RuntimeError): verify_batch(p, ['a', 'b'], 'identity', 2, 8)

    def test_uncommitted_data_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'batch_000000.npz'
            np.savez(p, hidden=np.ones((1, 2, 8), np.float32), sample_ids=['a'])
            self.assertFalse(verify_batch(p, ['a'], 'identity', 2, 8))

    def test_stage_cannot_be_swapped_even_with_shared_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'batch_000000.npz'
            audit = dict(stage='base', mode='formal', protocol_digest='identity', fp32=True, tf32=False)
            commit_batch(p, ['a'], np.ones((1, 2, 8), np.float32), 'identity', audit)
            self.assertTrue(verify_batch(p, ['a'], 'identity', 2, 8, stage='base', mode='formal'))
            with self.assertRaises(RuntimeError): verify_batch(p, ['a'], 'identity', 2, 8, stage='sft', mode='formal')
            with self.assertRaises(RuntimeError): verify_batch(p, ['a'], 'identity', 2, 8, stage='base', mode='pilot')


class HookTests(unittest.TestCase):
    def test_raw_blocks_last_valid_token_and_final_norm(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaModel
        except ImportError:
            self.skipTest('Install requirements.txt to run the tiny-model extraction test')
        torch.manual_seed(17)
        cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
                          attention_dropout=0.0)
        cfg._attn_implementation = 'eager'
        model = LlamaModel(cfg).eval().requires_grad_(False)
        capture = LastTokenCapture(model)
        try:
            with torch.inference_mode():
                short, error = capture.run(torch.tensor([[3, 5, 7]]), torch.ones((1, 3), dtype=torch.long), True)
                padded, _ = capture.run(torch.tensor([[3, 5, 7, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]), True)
                np.testing.assert_allclose(short, padded, rtol=1e-5, atol=1e-6)
                self.assertEqual(short.shape, (32, 16)); self.assertLessEqual(error, 1e-6)
                direct = model(torch.tensor([[3, 5, 7]]), output_hidden_states=True, use_cache=False)
                np.testing.assert_allclose(short[0], direct.hidden_states[1][0, -1].numpy())
                self.assertFalse(np.allclose(short[-1], direct.last_hidden_state[0, -1].numpy()))
        finally:
            capture.close()


if __name__ == '__main__': unittest.main()


def test_native_template_and_plain_readout_on_tiny_model(tmp_path):
    """Use an actual local tokenizer loader and the published Tulu template."""
    import json
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaModel
    from tulu_id.tokenize import load_native_tokenizer, encode_item
    from tulu_id.common import ROOT
    template = json.loads((ROOT / 'provenance/tokenization.json').read_text())['stages']['sft']['effective_chat_template']
    backend = Tokenizer(WordLevel({'[UNK]':0, '<|user|>':1, '<|assistant|>':2, '</s>':3,
                                  'A':4, 'fact':5, 'Is':6, 'it':7, 'true':8, '?':9}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='</s>',
                                       additional_special_tokens=['<|user|>','<|assistant|>'])
    tokenizer.chat_template = template
    tokenizer.save_pretrained(tmp_path)
    native = load_native_tokenizer(tmp_path)
    item = dict(sample_id='tiny', question='A fact', query='Is it true?', text='A fact Is it true?')
    torch.manual_seed(19)
    cfg = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1)
    cfg._attn_implementation = 'eager'
    model = LlamaModel(cfg).eval().requires_grad_(False)
    capture = LastTokenCapture(model)
    try:
        for wrapper in ['plain','native']:
            row = encode_item(native, item, wrapper)
            assert row['readout_index'] == len(row['input_ids']) - 1
            if wrapper == 'native':
                assert row['input_ids'][-1] == native.convert_tokens_to_ids('<|assistant|>')
            ids = torch.tensor([row['input_ids']])
            mask = torch.tensor([row['attention_mask']])
            with torch.inference_mode():
                states, error = capture.run(ids, mask, check_norm=True)
                again, _ = capture.run(ids, mask)
                direct = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                np.testing.assert_array_equal(states, again)
                for layer in range(31):
                    np.testing.assert_allclose(states[layer], direct.hidden_states[layer+1][0,-1].numpy(), rtol=1e-6, atol=1e-7)
                assert states.shape == (32,16) and error <= 1e-6
    finally:
        capture.close()
