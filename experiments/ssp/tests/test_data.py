from dataclasses import dataclass

from ssp_tulu.data import (
    encode_geometry_trajectory,
    encode_training_trajectory,
    generate_triples,
)


@dataclass
class TinyTokenizer:
    bos_token_id: int = 1
    eos_token_id: int = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + index for index, token in enumerate(text.split()) if token]


def test_training_encoding_only_truncates_at_step_boundary():
    tokenizer = TinyTokenizer()
    encoded = encode_training_trajectory(
        tokenizer,
        "question words",
        ["one two", "three four", "five six seven"],
        step_token_id=99,
        max_length=11,
    )
    assert encoded is not None
    assert encoded.steps_kept == 2
    assert len(encoded.boundary_positions) == 3
    assert encoded.input_ids[encoded.boundary_positions[-1]] == 99
    assert all(value == -100 for value in encoded.labels[:4])


def test_geometry_natural_boundaries():
    tokenizer = TinyTokenizer()
    ids, boundaries = encode_geometry_trajectory(
        tokenizer,
        "q",
        ["a b", "c d"],
        max_length=32,
        boundary_mode="natural",
        step_token_id=99,
        literal_marker="<|step|>",
    )
    assert len(boundaries) == 3
    assert boundaries[-1] == len(ids) - 1


def test_triple_strategies_are_deterministic_and_count_matched():
    boundaries = [3, 7, 11, 15, 19]
    a = generate_triples("A", 25, boundaries, "record", 42)
    c_match = generate_triples("C-match", 25, boundaries, "record", 42)
    a2_match = generate_triples("A2-match", 25, boundaries, "record", 42)
    assert len(a) == len(c_match) == len(a2_match) == 3
    assert c_match == generate_triples("C-match", 25, boundaries, "record", 42)
    assert all(triple[0] < triple[1] < triple[2] for triple in c_match + a2_match)
