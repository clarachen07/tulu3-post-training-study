from ssp_tulu.segmentation import raw_paragraph_steps, semantic_clean_steps, split_math_steps


def test_paragraph_first_keeps_asy_together():
    text = "First derive x.\n\n[asy]\ndraw((0,0)--(1,1));\n\nlabel(\"x\",(1,1));\n[/asy]\n\nTherefore x=1."
    result = split_math_steps(text)
    assert result.mode == "paragraph"
    assert len(result.steps) == 3
    assert "[asy]" in result.steps[1]


def test_sentence_fallback():
    result = split_math_steps("First compute x. Therefore x is two.")
    assert result.mode == "sentence_fallback"
    assert len(result.steps) == 2


def test_raw_paragraphs_do_not_fall_back_to_sentences():
    result = raw_paragraph_steps("First compute x. Therefore x is two.")
    assert result.mode == "paragraph"
    assert len(result.steps) == 1


def test_semantic_clean_merges_short_fragment():
    result = semantic_clean_steps("We compute the long first expression here.\n\nThus.\n\nThe answer follows.")
    assert len(result.steps) == 2
    assert "Thus." in result.steps[0]
