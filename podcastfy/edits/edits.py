import logging
from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing import List
import logging

import re
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
    rationale: str = Field(
        default="",
        description="The rationale or reason for the edit."
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
                    "rationale: "This line provides additional context.",
                    "text": "This is the newly added line."
                },
                {
                    "action": "replace",
                    "line_number": 10,
                    "rationale: "This line provides additional context.",
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
    edits_data = load_json_agnostic_to_quotes(cleaned_json)

    response = EditsResponse(edits=edits_data)
    return convert_edits_response_to_models(response)

def convert_edits_response_to_models(response: EditsResponse) -> List[EditModel]:
    edit_models = []
    for edit_dict in response.edits:
        try:
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
        except Exception as e:
            logger.error(f"Error converting edit: {edit_dict}")
            logger.error(e)

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

def load_json_agnostic_to_quotes(json_like_string):
    # Check if the input is already valid JSON
    try:
        return json.loads(json_like_string)
    except json.JSONDecodeError:
        pass  # If this fails, we attempt preprocessing
    
    # Replace single quotes with double quotes, but cautiously
    processed_string = re.sub(
        r"(?<![\\\"'])'(?![\\\"'])", '"',  # Replace single quotes not surrounded by valid JSON syntax
        json_like_string
    )
    
    try:
        # Attempt to load the processed string
        return json.loads(processed_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse JSON-like string: {e}")

def edit_transcript(
    transcript: str,
    instruction: str,
    llm: BaseChatModel
) -> str:
    """
    Edits a podcast transcript by fixing grammar errors and ensuring a clean flow between speakers.

    Parameters:
        transcript (str): The original transcript to be edited.
        instruction (str, optional): The prompt sentence guiding the edits.
                                    Example:
                                    "Suggest other edits to the transcript to fix any grammar errors. 
                                     Strive to make a clean flow make sense from speaker to speaker."

    Returns:
        str: The final edited transcript.
    """
    # Add line numbers to the transcript
    transcript_with_lines: str = add_line_numbers(transcript=transcript).strip()

    # Define the prompt template with the configurable suggestion
    edit_prompt_template: str = """
You are a podcast editor. 
{instruction}

Example:
    [
        {{
            "action": "addition",
            "line_number": 5,
            "rationale: "This line provides additional context.",
            "text": "This is the newly added line."
        }},
        {{
            "action": "replace",
            "line_number": 10,
            "rationale: "This line provides additional context.",
            "text": "Replace the content of line 10 with this text."
        }},
        {{
            "action": "replace",
            "line_number": 13,
            "rationale: "This line provides additional context.",
            "text": ""
        }}
    ]
    
Each dictionary in the `edits` list should contain:
    - action: A string, either "addition" or "replace".
    - line_number: An integer specifying the target line (1-based).
    - rationale: A string explaining the reason for the edit.
    - text: A string representing the text to add or replace at the specified line.
            For "addition", this text is inserted at the given line.
            For "replace", this text replaces the line. If text is empty for "replace",
            it effectively deletes that line.

Transcript:
{transcript}

List of edits of deletions and additions (no comments or preamble):
"""

    # Create a PromptTemplate instance with the provided template
    edit_prompt: PromptTemplate = PromptTemplate(
        input_variables=["transcript"],
        template=edit_prompt_template
    )

    # Combine the prompt template with the language model (llm)
    edit_chain = edit_prompt | llm | StrOutputParser()

    # Log the execution of the edit chain
    logger.debug("Executing edit chain")

    # Invoke the language model with the transcript to get edits
    edit_str: str = edit_chain.invoke({"transcript": transcript_with_lines, "instruction": instruction})

    # Convert the raw edit response into structured edit models
    edits: List[EditModel] = convert_edits_response_to_models_raw(edit_str)
    
    # Log edits
    logger.debug(f"{len(edits)} edits proposed.") 
    for edit in edits:
        logger.debug(f"Edit: {edit}")

    # Prepare the transcript lines for editing
    original_lines: List[str] = [
        line.split(":", 1)[1].strip() for line in transcript_with_lines.split("\n") if ":" in line
    ]

    # Apply the edits to the original transcript lines
    transcript_edited: List[str] = apply_edits(original_lines, edits)

    # Join the edited lines into the final transcript
    final_transcript: str = "\n".join(transcript_edited).strip()

    return final_transcript
