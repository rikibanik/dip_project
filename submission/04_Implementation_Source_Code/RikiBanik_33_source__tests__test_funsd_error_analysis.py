from scripts.run_funsd_error_analysis import _order_inversions,_sequence_counts,_word_error_categories

def test_character_operation_categories():
    counts,_=_sequence_counts("cat", "cut")
    assert counts=={"insertions":0,"deletions":0,"substitutions":1}
    counts,_=_sequence_counts("cat", "cats")
    assert counts=={"insertions":1,"deletions":0,"substitutions":0}
    counts,_=_sequence_counts("cats", "cat")
    assert counts=={"insertions":0,"deletions":1,"substitutions":0}

def test_word_segmentation_detects_exact_split():
    analysis=_word_error_categories("helloworld", "hello world")
    assert analysis["counts"]["word_segmentation_errors"]==1
    assert analysis["counts"]["extra_words"]==0
    assert analysis["segmentation_examples"][0]["type"]=="one_reference_word_split_in_ocr"

def test_order_inversion_counts_reordered_shared_tokens():
    assert _order_inversions(["alpha","beta","gamma"],["gamma","alpha","beta"])==2

