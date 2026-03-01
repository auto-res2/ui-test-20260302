"""Dataset loading and preprocessing for GSM8K."""

import re
from pathlib import Path
from datasets import load_dataset


def load_gsm8k(split: str, cache_dir: str = ".cache") -> list[dict]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split specification (e.g., "train[:50]", "test[:200]")
        cache_dir: Directory for caching downloaded datasets

    Returns:
        List of examples with 'question' and 'answer' fields
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    examples = []
    for item in dataset:
        # Extract numeric answer from the formatted answer string
        answer_text = item["answer"]
        # GSM8K answers are formatted as: "step1\nstep2\n#### numeric_answer"
        numeric_answer = extract_numeric_answer(answer_text)

        examples.append(
            {
                "question": item["question"],
                "answer": numeric_answer,
                "full_answer": answer_text,
            }
        )

    return examples


def extract_numeric_answer(text: str) -> float:
    """
    Extract numeric answer from GSM8K answer format.
    GSM8K uses "#### number" to denote the final answer.

    Args:
        text: Answer text containing "#### number"

    Returns:
        Numeric answer as float
    """
    # Look for #### followed by a number
    match = re.search(r"####\s*([-+]?[\d,]+\.?\d*)", text)
    if match:
        # Remove commas from numbers (e.g., "1,234" -> "1234")
        number_str = match.group(1).replace(",", "")
        return float(number_str)

    # Fallback: try to find any number in the text
    numbers = re.findall(r"[-+]?[\d,]+\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    raise ValueError(f"Could not extract numeric answer from: {text}")


def extract_answer_from_response(response: str) -> float:
    """
    Extract numeric answer from model's chain-of-thought response.

    Args:
        response: Model's generated response

    Returns:
        Extracted numeric answer as float
    """
    # Try common answer patterns
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([-+]?[\d,]+\.?\d*)",
        r"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer\s+is\s*:?\s*)?([-+]?[\d,]+\.?\d*)",
        r"=\s*([-+]?[\d,]+\.?\d*)\s*$",  # Ends with "= number"
        r"\$\s*([-+]?[\d,]+\.?\d*)",  # Dollar amounts
        r"####\s*([-+]?[\d,]+\.?\d*)",  # GSM8K format
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            number_str = match.group(1).replace(",", "")
            return float(number_str)

    # Fallback: extract last number in response
    numbers = re.findall(r"[-+]?[\d,]+\.?\d*", response)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    raise ValueError(
        f"Could not extract numeric answer from response: {response[:200]}"
    )
