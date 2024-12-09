"""
Podcastfy CLI

This module provides a command-line interface for generating podcasts or transcripts
from URLs or existing transcript files. It orchestrates the content extraction,
generation, and text-to-speech conversion processes.
"""

import os
import uuid
import typer
import yaml
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.content_generator import ContentGenerator, LLMBackend
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.config import Config, load_config
from podcastfy.utils.config_conversation import load_conversation_config
from podcastfy.utils.logger import setup_logger
from typing import List, Optional, Dict, Any, Tuple
import copy
from dataclasses import dataclass

# Import necessary classes for structured output parsing
from pydantic import BaseModel, Field


import logging

# Configure logging to show all levels and write to both file and console
""" logging.basicConfig(
    level=logging.DEBUG,  # Show all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcastfy.log'),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
) """


logger = setup_logger(__name__)

app = typer.Typer()

os.environ["LANGCHAIN_TRACING_V2"] = "False"


@dataclass
class ProcessedContent:
    """
    Represents the result of the content processing pipeline.

    Attributes:
        transcript_file_path (Optional[str]): Path to the generated transcript file, if any.
        urls (Optional[List[str]]): List of URLs that were processed, if applicable.
        audio_file_path (Optional[str]): Path to the generated audio file, if any.
        title (Optional[str]): Generated title based on the transcript or processed content.
        description (Optional[str]): Generated description based on the transcript or processed content.
    """

    transcript_file_path: Optional[str]
    urls: Optional[List[str]]
    audio_file_path: Optional[str]
    title: Optional[str]
    description: Optional[str]

def detect_and_unescape(string):
    """
    Detects and unescapes escape sequences in a given string.

    Args:
        string (str): The input string potentially containing escape sequences.

    Returns:
        str: The unescaped version of the string if it contains escape sequences, 
             otherwise the original string.
    """
    try:
        # Check if there are any escape sequences by comparing with unescaped version
        unescaped_string = string.encode('utf-8').decode('unicode_escape')
        return unescaped_string
    except (UnicodeDecodeError, AttributeError):
        # If decoding fails or the string is already plain, return it as is
        return string
    
def generate_description_and_title(
    transcript: str,
    config: Dict[str, Any],
    is_local: bool = False,
    model_name: str = "gemini-1.5-pro-latest",
    api_key_label: str = "GEMINI_API_KEY",
    conversation_config: Optional[Dict[str, Any]] = None,
    generate_title: bool = True,
) -> Tuple[Optional[str], str]:
    # Simulate using LangChain for title and description generation with structured output
    if generate_title:
        logger.info("Generating podcast title and description.")
    else:
        logger.info("Generating podcast description.")

    config_conversation = load_conversation_config(conversation_config)
    backend = LLMBackend(
        is_local=is_local,
        temperature=config_conversation.get("creativity", 1),
        max_output_tokens=8192,
        model_name=model_name,
        api_key_label=api_key_label,
    )

    llm = backend.llm

    # Define the schema using Pydantic
    if generate_title:

        class PodcastOutput(BaseModel):
            """Podcast title and description."""

            title: str = Field(description="The podcast title")
            description: str = Field(description="The podcast description")

    else:

        class PodcastOutput(BaseModel):
            """Podcast description."""

            description: str = Field(description="The podcast description")

    # Create an LLM with structured output
    structured_llm = llm.with_structured_output(PodcastOutput)

    # Prepare the prompt
    if generate_title:
        what_to_generate = (
            "Extract the original article title of the source material (do not include the podcast name) and generate a 3-paragraph description"
        )
    else:
        what_to_generate = "a 3-paragraph description"

    prompt_template_str = config.get(
        "podcast_generation_prompt",
        (
            "You are a podcast editor. Analyze the following podcast transcript and create {what_to_generate} for it, "
            "noting the date, author, and topic.\n\nDo not escape quotation marks or special characters.\n\n"
            "Transcript:\n{transcript}"
        ),
    )

    prompt = PromptTemplate(
        input_variables=["transcript", "what_to_generate"],
        template=prompt_template_str,
    )

    # Create an LLM chain
    chain = prompt | structured_llm

    # Run the chain to get structured output
    output = chain.invoke(
        {
            "transcript": transcript,
            "what_to_generate": what_to_generate,
        }
    )

    # Extract title and description from the output
    if generate_title:
        title = detect_and_unescape(output.title)
        description = detect_and_unescape(output.description)
        logger.info(f"Generated podcast title: {title}")
    else:
        title = None
        description = detect_and_unescape(output.description)

    logger.info(f"Generated podcast description: {description}")

    return title, description


def process_content(
    urls: Optional[List[str]] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    generate_audio: bool = True,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
) -> ProcessedContent:
    """
    Process URLs, a transcript file, image paths, or raw text to generate a podcast or transcript,
    returning a dataclass with transcript, URLs, audio file path, and a generated description.
    """
    try:
        if config is None:
            config = load_config()

        conv_config = load_conversation_config()
        if conversation_config:
            conv_config.configure(conversation_config)

        tts_config = conv_config.get("text_to_speech", {})
        output_directories = tts_config.get("output_directories", {})

        qa_content = ""
        if transcript_file:
            logger.info(f"Using transcript file: {transcript_file}")
            with open(transcript_file, "r") as file:
                qa_content = file.read()
        else:
            content_extractor = (
                ContentExtractor()
                if urls or topic or (text and longform and len(text.strip()) < 100)
                else None
            )
            content_generator = ContentGenerator(
                is_local=is_local,
                model_name=model_name,
                api_key_label=api_key_label,
                conversation_config=conv_config.to_dict(),
            )

            combined_content = ""
            if urls:
                logger.info(f"Processing {len(urls)} links")
                contents = [content_extractor.extract_content(link) for link in urls]
                combined_content += "\n\n".join(contents)

            if text:
                if longform and len(text.strip()) < 100:
                    expanded_content = content_extractor.generate_topic_content(text)
                    combined_content += f"\n\n{expanded_content}"
                else:
                    combined_content += f"\n\n{text}"

            if topic:
                topic_content = content_extractor.generate_topic_content(topic)
                combined_content += f"\n\n{topic_content}"

            random_filename = f"transcript_{uuid.uuid4().hex}.txt"
            transcript_filepath = os.path.join(
                output_directories.get("transcripts", "data/transcripts"),
                random_filename,
            )
            qa_content = content_generator.generate_qa_content(
                combined_content,
                image_file_paths=image_paths or [],
                output_filepath=transcript_filepath,
                longform=longform,
            )

        audio_file_path = None
        if generate_audio:
            api_key = (
                None
                if tts_model == "edge"
                else getattr(
                    config, f"{tts_model.upper().replace('MULTI', '')}_API_KEY"
                )
            )

            text_to_speech = TextToSpeech(
                model=tts_model,
                api_key=api_key,
                conversation_config=conv_config.to_dict(),
            )
            random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
            audio_file_path = os.path.join(
                output_directories.get("audio", "data/audio"), random_filename
            )
            text_to_speech.convert_to_speech(qa_content, audio_file_path)
            logger.info(f"Podcast generated successfully using {tts_model} TTS model")

        title, description = generate_description_and_title(
            transcript=qa_content,
            config=config,
            is_local=is_local,
            model_name=model_name,
            api_key_label=api_key_label,
            conversation_config=conversation_config,
        )

        return ProcessedContent(
            transcript_file_path=transcript_filepath,
            urls=urls,
            audio_file_path=audio_file_path,
            title=title,
            description=description,
        )

    except Exception as e:
        logger.error(f"An error occurred in the process_content function: {str(e)}")
        raise


@app.command()
def main(
    urls: list[str] = typer.Option(None, "--url", "-u", help="URLs to process"),
    file: typer.FileText = typer.Option(
        None, "--file", "-f", help="File containing URLs, one per line"
    ),
    transcript: typer.FileText = typer.Option(
        None, "--transcript", "-t", help="Path to a transcript file"
    ),
    tts_model: str = typer.Option(
        None,
        "--tts-model",
        "-tts",
        help="TTS model to use (openai, elevenlabs, edge, or gemini)",
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Generate only a transcript without audio"
    ),
    conversation_config_path: str = typer.Option(
        None,
        "--conversation-config",
        "-cc",
        help="Path to custom conversation configuration YAML file",
    ),
    image_paths: List[str] = typer.Option(
        None, "--image", "-i", help="Paths to image files to process"
    ),
    is_local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use a local LLM instead of a remote one (http://localhost:8080)",
    ),
    text: str = typer.Option(
        None, "--text", "-txt", help="Raw text input to be processed"
    ),
    llm_model_name: str = typer.Option(
        None, "--llm-model-name", "-m", help="LLM model name for transcript generation"
    ),
    api_key_label: str = typer.Option(
        None, "--api-key-label", "-k", help="Environment variable name for LLMAPI key"
    ),
    topic: str = typer.Option(
        None, "--topic", "-tp", help="Topic to generate podcast about"
    ),
    longform: bool = typer.Option(
        False,
        "--longform",
        "-lf",
        help="Generate long-form content (only available for text input without images)",
    ),
):
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, image files, or raw text.
    """
    try:
        config = load_config()
        main_config = config.get("main", {})

        conversation_config = None
        # Load conversation config if provided
        if conversation_config_path:
            with open(conversation_config_path, "r") as f:
                conversation_config: Dict[str, Any] | None = yaml.safe_load(f)

        # Use default TTS model from conversation config if not specified
        if tts_model is None:
            tts_config = load_conversation_config().get("text_to_speech", {})
            tts_model = tts_config.get("default_tts_model", "openai")

        if transcript:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            final_output = process_content(
                transcript_file=transcript.name,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                conversation_config=conversation_config,
                config=config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
            )
        else:
            urls_list = urls or []
            if file:
                urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise typer.BadParameter(
                    "No input provided. Use --url, --file, --transcript, --image, --text, or --topic."
                )

            final_output = process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
            )

        if transcript_only:
            typer.echo(f"Transcript generated successfully: {final_output}")
        else:
            typer.echo(
                f"Podcast generated successfully using {tts_model} TTS model: {final_output}"
            )

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


def generate_podcast(
    urls: Optional[List[str]] = None,
    url_file: Optional[str] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    transcript_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
) -> Optional[ProcessedContent]:
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, or image files.

    Args:
        urls (Optional[List[str]]): List of URLs to process.
        url_file (Optional[str]): Path to a file containing URLs, one per line.
        transcript_file (Optional[str]): Path to a transcript file.
        tts_model (Optional[str]): TTS model to use ('openai' [default], 'elevenlabs', 'edge', or 'gemini').
        transcript_only (bool): Generate only a transcript without audio. Defaults to False.
        config (Optional[Dict[str, Any]]): User-provided configuration dictionary.
        conversation_config (Optional[Dict[str, Any]]): User-provided conversation configuration dictionary.
        image_paths (Optional[List[str]]): List of image file paths to process.
        is_local (bool): Whether to use a local LLM. Defaults to False.
        text (Optional[str]): Raw text input to be processed.
        llm_model_name (Optional[str]): LLM model name for content generation.
        api_key_label (Optional[str]): Environment variable name for LLM API key.
        topic (Optional[str]): Topic to generate podcast about.

    Returns:
        Optional[str]: Path to the final podcast audio file, or None if only generating a transcript.
    """
    try:
        print("Generating podcast...")
        # Load default config
        default_config = load_config()

        # Update config if provided
        if config:
            if isinstance(config, dict):
                # Create a deep copy of the default config
                updated_config = copy.deepcopy(default_config)
                # Update the copy with user-provided values
                updated_config.configure(**config)
                default_config = updated_config
            elif isinstance(config, Config):
                # If it's already a Config object, use it directly
                default_config = config
            else:
                raise ValueError(
                    "Config must be either a dictionary or a Config object"
                )

        if not conversation_config:
            conversation_config = load_conversation_config().to_dict()

        main_config = default_config.config.get("main", {})

        # Use provided tts_model if specified, otherwise use the one from config
        if tts_model is None:
            tts_model = conversation_config.get("default_tts_model", "openai")

        if transcript_file:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            return process_content(
                transcript_file=transcript_file,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
            )
        else:
            urls_list = urls or []
            if url_file:
                with open(url_file, "r") as file:
                    urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise ValueError(
                    "No input provided. Please provide either 'urls', 'url_file', "
                    "'transcript_file', 'image_paths', 'text', or 'topic'."
                )

            return process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
