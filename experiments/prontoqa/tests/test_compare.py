"""Scientific equality rejects changed counts and mismatched pairings."""
import pytest
from tulu_id.compare import compare_tables


def test_reordered_pairs_and_small_float_roundoff_are_allowed(tmp_path):
    left,right = tmp_path/'reference.csv',tmp_path/'reproduction.csv'
    left.write_text('stage,replicate,pc_id90,id\nbase,0,10,3.0\nbase,1,11,4.0\n')
    right.write_text('stage,replicate,pc_id90,id\nbase,1,11,4.000000000001\nbase,0,10,3.0\n')
    assert compare_tables(left,right)['rows']==2
    right.write_text('stage,replicate,pc_id90,id\nbase,0,11,3.0\nbase,1,11,4.0\n')
    with pytest.raises(ValueError,match='Numeric reproduction differs'):
        compare_tables(left,right)


def test_duplicate_coordinates_cannot_replace_a_missing_replicate(tmp_path):
    left,right = tmp_path/'reference.csv',tmp_path/'reproduction.csv'
    left.write_text('stage,replicate,id\nbase,0,3\nbase,1,4\n')
    right.write_text('stage,replicate,id\nbase,0,3\nbase,0,4\n')
    with pytest.raises(ValueError,match='Duplicate comparison coordinates'):
        compare_tables(left,right)
