import logging
from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EditAction(str, Enum):
    ADDITION = "addition"
    REPLACE = "replace"

class EditModel(BaseModel):
    action: EditAction = Field(
        description="The type of edit action (addition or replace)."
    )
    line_number: int = Field(
        description="The line number where the edit is to be applied (1-based)."
    )
    text: str = Field(
        default="",
        description="The text to use for addition or replacement. If empty on replace, it's effectively a deletion."
    )

class EditsResponse(BaseModel):
    """
    Represents a response containing a list of edits to be applied.
    
    Example:
        EditsResponse(
            edits=[
                {
                    "action": "addition",
                    "line_number": 5,
                    "text": "This is the newly added line."
                },
                {
                    "action": "replace",
                    "line_number": 10,
                    "text": "Replace the content of line 10 with this text."
                }
            ]
        )
    """
    edits: List[Dict[str, Any]] = Field(
        description="The list of edits to apply, each represented as a dictionary."
    )

def add_line_numbers(transcript: str) -> str:
    return "\n".join([f"{index+1}: {line}" for index, line in enumerate(transcript.split("\n"))])

def clean_text(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned.strip()

def convert_edits_response_to_models_raw(raw_json: str) -> List[EditModel]:
    cleaned_json = clean_text(raw_json)
    edits_data = json.loads(cleaned_json)

    response = EditsResponse(edits=edits_data)
    return convert_edits_response_to_models(response)

def convert_edits_response_to_models(response: EditsResponse) -> List[EditModel]:
    edit_models = []
    for edit_dict in response.edits:
        action_str = edit_dict.get("action")
        line_num_raw = edit_dict.get("line_number")
        
        if line_num_raw is None:
            raise KeyError("Missing required field 'line_number'")
        try:
            line_num = int(line_num_raw)
        except (TypeError, ValueError):
            raise ValueError(f"line_number must be an integer, got {line_num_raw}")
        
        text_str = edit_dict.get("text", "")
        text_str = clean_text(text_str)

        if action_str not in ("addition", "replace"):
            raise ValueError(f"Invalid action '{action_str}' provided.")

        edit_model = EditModel(
            action=action_str,
            line_number=line_num,
            text=text_str
        )
        edit_models.append(edit_model)

    return edit_models

def apply_edits(lines: List[str], edits: List[EditModel]) -> List[str]:
    # Represent final lines as a sorted list of (line_number, text).
    # original lines start as (1, line1), (2, line2), ...
    final_lines = [(i+1, l) for i, l in enumerate(lines)]

    # Sort edits by line_number ascending
    edits_sorted = sorted(edits, key=lambda e: e.line_number)

    for edit in edits_sorted:
        line_num = edit.line_number
        action = edit.action
        text = edit.text

        # Find the position of line_num in final_lines
        # We perform a binary-like search to find insertion/replacement point
        idx = None
        for i, (ln, _) in enumerate(final_lines):
            if ln == line_num:
                idx = i
                break

        if action == EditAction.ADDITION:
            # Insert a new line at the position corresponding to line_num
            if idx is not None:
                # If that exact line_num exists, insert before it
                final_lines.insert(idx, (line_num, text))
            else:
                # If not found, find where it would fit
                insertion_point = None
                for i, (ln, _) in enumerate(final_lines):
                    if ln > line_num:
                        insertion_point = i
                        break
                if insertion_point is None:
                    # No larger line_num found, append at the end
                    final_lines.append((line_num, text))
                else:
                    # Insert before the next larger line_num
                    final_lines.insert(insertion_point, (line_num, text))

        elif action == EditAction.REPLACE:
            # Replace or delete
            if text == "":
                # Deletion
                if idx is not None:
                    del final_lines[idx]
                # If line doesn't exist, it's a no-op
            else:
                # Replacement
                if idx is None:
                    # Line doesn't exist, cannot replace
                    raise ValueError(f"Cannot replace line {line_num}: line does not exist.")
                else:
                    # Replace the text
                    final_lines[idx] = (line_num, text)
        else:
            raise ValueError(f"Unknown action {action}")

    # final_lines is always kept sorted, no additional sort needed
    updated_texts = [t for (_, t) in final_lines]
    return updated_texts
