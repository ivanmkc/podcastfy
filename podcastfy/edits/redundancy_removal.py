from abc import ABC, abstractmethod
from typing import List, Sequence
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class EmbeddingFunction(ABC):
    """
    Abstract base class for embedding functions.
    Defines the contract for generating embeddings for a list of sentences.
    """

    @abstractmethod
    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences (List[str]): A list of sentences to embed.
        
        Returns:
            np.ndarray: A 2D array of embeddings.
        """
        pass


@dataclass
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function implementation using Google's Gemini API.
    """

    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        logging.debug(f"Generating embeddings for {len(sentences)} sentences.")
        return np.array([
            genai.embed_content(model="models/text-embedding-004", content=sentence)["embedding"]
            for sentence in sentences
        ])


class StopCondition(ABC):
    """
    Abstract base class for stop conditions.
    Defines the contract for determining when to stop the redundancy reduction process.
    """

    @abstractmethod
    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        """
        Determine whether the redundancy reduction process should stop.
        
        Args:
            sentences (List[str]): The current list of sentences.
            embeddings (np.ndarray): The current embeddings of the sentences.
            normalized_similarity_matrix (np.ndarray): The normalized similarity matrix of sentences.
        
        Returns:
            bool: True if the process should stop, False otherwise.
        """
        pass


@dataclass
class CombinedStopCondition(StopCondition):
    """
    Combines multiple stop conditions. Stops if any condition signals to stop.
    """
    conditions: Sequence[StopCondition]

    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        for condition in self.conditions:
            if condition.should_stop(sentences, embeddings, normalized_similarity_matrix):
                logging.debug(f"Stop condition met by: {condition.__class__.__name__}")
                return True
        return False


@dataclass
class ElbowStopCondition(StopCondition):
    """
    Stop condition based on detecting the "elbow" point in redundancy scores.
    """

    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        from kneed import KneeLocator

        mean_scores = normalized_similarity_matrix.sum(axis=1).mean()
        logging.debug(f"Mean normalized redundancy score: {mean_scores}")

        if len(normalized_similarity_matrix) > 2:
            elbow_detector = KneeLocator(
                range(len(normalized_similarity_matrix)),
                normalized_similarity_matrix.sum(axis=1),
                curve="convex",
                direction="decreasing"
            )
            if elbow_detector.knee is not None:
                logging.debug(f"Elbow detected at iteration {len(normalized_similarity_matrix)}.")
                return True
        return False


@dataclass
class ThresholdStopCondition(StopCondition):
    """
    Stop condition based on a similarity score threshold.
    """
    threshold: float = 0.9  # Default threshold for similarity

    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        max_similarity = normalized_similarity_matrix.max()
        logging.debug(f"Maximum normalized similarity: {max_similarity}. Threshold: {self.threshold}.")
        return max_similarity < self.threshold


@dataclass
class MinSentencesStopCondition(StopCondition):
    """
    Stop condition based on a minimum number of sentences.
    """
    min_sentences: int = 3

    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        current_count = len(sentences)
        logging.debug(f"Current sentence count: {current_count}. Minimum required: {self.min_sentences}.")
        return current_count <= self.min_sentences


@dataclass
class MinDeletionRatioStopCondition(StopCondition):
    """
    Stop condition based on the minimum deletion ratio.
    """
    min_ratio: float  # Minimum fraction of sentences to delete (e.g., 0.5)

    def should_stop(
        self, sentences: List[str], embeddings: np.ndarray, normalized_similarity_matrix: np.ndarray
    ) -> bool:
        original_count = embeddings.shape[0] + len(sentences)
        deleted_ratio = (original_count - len(sentences)) / original_count
        logging.debug(f"Current deletion ratio: {deleted_ratio:.2f}. Minimum required: {self.min_ratio}.")
        return deleted_ratio >= self.min_ratio


def strip_tags(sentence: str) -> str:
    """
    Removes tags like <Person1> or <Person2> from a sentence.
    
    Args:
        sentence (str): Input sentence with tags.
    
    Returns:
        str: Sentence without tags.
    """
    return re.sub(r"<[^>]+>", "", sentence).strip()


def remove_redundant_sentences(
    text: str,
    stop_condition: StopCondition,
    embedding_function: EmbeddingFunction
) -> str:
    """
    Removes redundant sentences iteratively based on cosine similarity.
    
    Args:
        text (str): Input text (single string), with sentences separated by newlines.
        stop_condition (StopCondition): An instance of a StopCondition subclass to determine when to stop.
        embedding_function (EmbeddingFunction): An instance of an EmbeddingFunction subclass to generate embeddings.
    
    Returns:
        str: Reconstituted text with redundant sentences removed.
    """
    logging.info("Starting redundancy reduction process.")
    sentences = [line.strip() for line in text.split("\n") if line.strip()]
    logging.debug(f"Split text into {len(sentences)} sentences.")

    if len(sentences) < 2:
        logging.warning("Text contains less than two sentences. No redundancy reduction performed.")
        return text

    stripped_sentences = [strip_tags(sentence) for sentence in sentences if strip_tags(sentence)]
    valid_sentences_map = [sentence for sentence in sentences if strip_tags(sentence)]

    if not stripped_sentences:
        logging.warning("No valid sentences after stripping tags. Returning original text.")
        return text

    embeddings = embedding_function.generate_embeddings(stripped_sentences)

    while True:
        # Compute cosine similarity and normalize to [0, 1]
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        normalized_similarity_matrix = (similarity_matrix + 1) / 2  # Normalize to [0, 1]

        # Check stop condition
        if stop_condition.should_stop(valid_sentences_map, embeddings, normalized_similarity_matrix):
            logging.info("Stop condition met. Terminating redundancy reduction.")
            break

        # Find the most redundant sentence (based on normalized similarity)
        redundancy_scores = normalized_similarity_matrix.sum(axis=1)
        most_redundant_idx = np.argmax(redundancy_scores)
        logging.debug(f"Removing sentence: {valid_sentences_map[most_redundant_idx]}")

        valid_sentences_map.pop(most_redundant_idx)
        embeddings = np.delete(embeddings, most_redundant_idx, axis=0)

    logging.info("Redundancy reduction completed.")
    return "\n".join(valid_sentences_map)
