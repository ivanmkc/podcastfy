import pytest

from typing import List
# from podcastfy.edits.edits import apply_edits, EditAction, EditModel, convert_edits_response_to_models, EditsResponse
# import pytest

from podcastfy.edits.edits import (
    EditAction,
    EditModel,
    EditsResponse,
    apply_edits,
    convert_edits_response_to_models,
    convert_edits_response_to_models_raw
)

def test_no_edits():
    lines = ["line1", "line2", "line3"]
    edits = []
    assert apply_edits(lines, edits) == lines

def test_replace_line():
    lines = ["line1", "line2", "line3"]
    edits = [EditModel(action=EditAction.REPLACE, line_number=2, text="NEW_LINE2")]
    updated = apply_edits(lines, edits)
    assert updated == ["line1", "NEW_LINE2", "line3"]

def test_delete_line():
    lines = ["line1", "line2", "line3"]
    edits = [EditModel(action=EditAction.REPLACE, line_number=2, text="")]
    updated = apply_edits(lines, edits)
    assert updated == ["line1", "line3"]

def test_add_line_in_middle():
    lines = ["line1", "line2", "line3"]
    edits = [EditModel(action=EditAction.ADDITION, line_number=2, text="inserted")]
    updated = apply_edits(lines, edits)
    # Insert at line_number=2 means put "inserted" before what was originally line2
    # Expected: line1, inserted, line2, line3
    assert updated == ["line1", "inserted", "line2", "line3"]

def test_add_line_at_end():
    lines = ["line1", "line2", "line3"]
    edits = [EditModel(action=EditAction.ADDITION, line_number=10, text="new_end")]
    updated = apply_edits(lines, edits)
    # line_number=10 does not exist and is greater than any line, so append
    assert updated == ["line1", "line2", "line3", "new_end"]

def test_delete_and_add_after():
    lines = ["line1", "line2", "line3", "line4"]
    # Delete line 2, then add line at line 3
    edits = [
        EditModel(action=EditAction.REPLACE, line_number=2, text=""),     # delete line 2
        EditModel(action=EditAction.ADDITION, line_number=3, text="new_line") # insert at line3 position
    ]
    updated = apply_edits(lines, edits)
    # After deletion of line2: final doc: line1, line3, line4
    # Insert at line3 => before original line3:
    # line1, new_line, line3, line4
    assert updated == ["line1", "new_line", "line3", "line4"]

def test_replace_nonexistent_line():
    lines = ["line1", "line2"]
    edits = [EditModel(action=EditAction.REPLACE, line_number=5, text="new_line5")]
    with pytest.raises(ValueError, match="does not exist"):
        apply_edits(lines, edits)

def test_delete_nonexistent_line_noop():
    lines = ["line1", "line2"]
    edits = [EditModel(action=EditAction.REPLACE, line_number=5, text="")]
    # Deleting a non-existent line should be no-op
    updated = apply_edits(lines, edits)
    assert updated == ["line1", "line2"]

def test_multiple_insertions():
    lines = ["line1", "line2", "line3"]
    edits = [
        EditModel(action=EditAction.ADDITION, line_number=1, text="before_line1"),
        EditModel(action=EditAction.ADDITION, line_number=2, text="before_line2"),
        EditModel(action=EditAction.ADDITION, line_number=10, text="at_end")
    ]
    updated = apply_edits(lines, edits)
    # Insert at line_number=1 => before line1: ["before_line1", "line1", "line2", "line3"]
    # Insert at line_number=2 => before line2:
    # ["before_line1", "line1", "before_line2", "line2", "line3"]
    # Insert at line_number=10 => at the end:
    # ["before_line1", "line1", "before_line2", "line2", "line3", "at_end"]
    assert updated == ["before_line1", "line1", "before_line2", "line2", "line3", "at_end"]
