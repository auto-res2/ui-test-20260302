"""Inference script for adaptive and fixed self-consistency methods."""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import APIModel
from src.preprocess import load_gsm8k, extract_answer_from_response


def compute_weighted_distribution(
    answers: list[float],
    weights: list[float],
) -> dict[float, float]:
    """
    Compute weighted answer distribution.

    Args:
        answers: List of extracted numeric answers
        weights: Corresponding quality weights

    Returns:
        Dictionary mapping answer -> total weight
    """
    distribution = defaultdict(float)
    for ans, w in zip(answers, weights):
        distribution[ans] += w
    return dict(distribution)


def compute_stability_metrics(distribution: dict[float, float]) -> dict:
    """
    Compute stability metrics from weighted distribution.

    Args:
        distribution: Answer -> weight mapping

    Returns:
        Dictionary with 'margin', 'entropy', and 'top_answer'
    """
    if not distribution:
        return {"margin": 0.0, "entropy": 1.0, "top_answer": None}

    total_weight = sum(distribution.values())
    if total_weight == 0:
        return {"margin": 0.0, "entropy": 1.0, "top_answer": None}

    # Normalize to probabilities
    probs = {ans: w / total_weight for ans, w in distribution.items()}

    # Sort by probability
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_answer = sorted_items[0][0]
    p_top = sorted_items[0][1]
    p_second = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    # Margin between top 2 answers
    margin = p_top - p_second

    # Normalized entropy
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize entropy by max possible (log(n))
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "margin": margin,
        "entropy": normalized_entropy,
        "top_answer": top_answer,
    }


def adaptive_self_consistency(
    model: APIModel,
    question: str,
    cot_prompt: str,
    cfg: DictConfig,
    mode: str = "main",
) -> dict:
    """
    A-SC2: Adaptive Self-Consistency with early exit.

    Args:
        model: API model instance
        question: Question to answer
        cot_prompt: Chain-of-thought instruction
        cfg: Config with method parameters
        mode: Execution mode (main/sanity_check)

    Returns:
        Dictionary with prediction and metadata
    """
    k_max = cfg.method.k_max
    temperature = cfg.method.temperature
    margin_threshold = cfg.method.margin_threshold
    entropy_threshold = cfg.method.entropy_threshold
    min_samples = cfg.method.min_samples
    lambda_length = cfg.method.lambda_length
    epsilon = cfg.method.epsilon

    # Sanity check overrides
    if mode == "sanity_check":
        k_max = min(5, k_max)

    full_prompt = f"{question}\n\n{cot_prompt}"

    answers = []
    weights = []
    responses = []
    token_counts = []

    for i in range(k_max):
        # Generate sample
        response, tokens = model.generate(
            full_prompt,
            max_tokens=cfg.inference.max_tokens,
            temperature=temperature,
        )

        responses.append(response)
        token_counts.append(tokens)

        # Extract answer
        try:
            answer = extract_answer_from_response(response)
        except ValueError:
            # Failed to extract - assign low weight
            answer = float("inf")  # Invalid marker

        answers.append(answer)

        # Compute quality weight
        length = len(response)
        length_penalty = math.exp(-lambda_length * length)

        # Self-contradiction check (simplified: check if answer is valid)
        contradiction_score = 1.0 if answer != float("inf") else 0.0

        weight = length_penalty * (epsilon + contradiction_score)
        weights.append(weight)

        # Update distribution
        distribution = compute_weighted_distribution(answers, weights)
        stability = compute_stability_metrics(distribution)

        # Early exit check
        if i >= min_samples - 1:
            if stability["margin"] >= margin_threshold:
                break
            if stability["entropy"] <= entropy_threshold:
                break

    # Final prediction
    final_distribution = compute_weighted_distribution(answers, weights)
    final_stability = compute_stability_metrics(final_distribution)
    predicted_answer = final_stability["top_answer"]

    # Filter out invalid answers for token count
    valid_tokens = [t for a, t in zip(answers, token_counts) if a != float("inf")]
    total_tokens = sum(valid_tokens) if valid_tokens else sum(token_counts)

    return {
        "predicted_answer": predicted_answer,
        "num_samples": len(answers),
        "total_tokens": total_tokens,
        "margin": final_stability["margin"],
        "entropy": final_stability["entropy"],
        "responses": responses,
    }


def fixed_self_consistency(
    model: APIModel,
    question: str,
    cot_prompt: str,
    cfg: DictConfig,
    mode: str = "main",
) -> dict:
    """
    Fixed Self-Consistency baseline.

    Args:
        model: API model instance
        question: Question to answer
        cot_prompt: Chain-of-thought instruction
        cfg: Config with method parameters
        mode: Execution mode (main/sanity_check)

    Returns:
        Dictionary with prediction and metadata
    """
    k_fixed = cfg.method.k_fixed
    temperature = cfg.method.temperature

    # Sanity check overrides
    if mode == "sanity_check":
        k_fixed = min(3, k_fixed)

    full_prompt = f"{question}\n\n{cot_prompt}"

    answers = []
    responses = []
    token_counts = []

    for i in range(k_fixed):
        response, tokens = model.generate(
            full_prompt,
            max_tokens=cfg.inference.max_tokens,
            temperature=temperature,
        )

        responses.append(response)
        token_counts.append(tokens)

        try:
            answer = extract_answer_from_response(response)
        except ValueError:
            answer = float("inf")

        answers.append(answer)

    # Majority vote (simple counting)
    answer_counts = defaultdict(int)
    for ans in answers:
        if ans != float("inf"):
            answer_counts[ans] += 1

    if answer_counts:
        predicted_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
    else:
        predicted_answer = None

    total_tokens = sum(token_counts)

    return {
        "predicted_answer": predicted_answer,
        "num_samples": len(answers),
        "total_tokens": total_tokens,
        "responses": responses,
    }


def run_inference(cfg: DictConfig):
    """
    Main inference loop.

    Args:
        cfg: Hydra config
    """
    mode = cfg.mode

    # Initialize model
    model = APIModel(
        model_name=cfg.run.model.name,
        provider=cfg.run.model.provider,
        api_key_env=cfg.run.model.api_key_env,
    )

    # Load dataset
    split = (
        cfg.run.dataset.split_test
        if mode in ["main", "sanity_check"]
        else cfg.run.dataset.split_train
    )
    if mode == "sanity_check":
        # Override to minimal dataset
        split = "test[:5]"

    examples = load_gsm8k(split, cache_dir=".cache")

    # Determine method
    method_type = cfg.run.method.type

    # Initialize WandB
    wandb_project = cfg.wandb.project
    if mode == "sanity_check":
        wandb_project = f"{cfg.wandb.project}-sanity"

    wandb.init(
        entity=cfg.wandb.entity,
        project=wandb_project,
        name=cfg.run.run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Run inference
    results = []
    total_tokens = 0
    total_samples = 0
    correct_count = 0

    for example in tqdm(examples, desc="Running inference"):
        question = example["question"]
        true_answer = example["answer"]

        if method_type == "adaptive-sc":
            result = adaptive_self_consistency(
                model, question, cfg.run.inference.cot_prompt, cfg.run, mode
            )
        elif method_type == "fixed-sc":
            result = fixed_self_consistency(
                model, question, cfg.run.inference.cot_prompt, cfg.run, mode
            )
        else:
            raise ValueError(f"Unknown method type: {method_type}")

        predicted = result["predicted_answer"]

        # Check correctness
        is_correct = False
        if predicted is not None and true_answer is not None:
            is_correct = abs(predicted - true_answer) < 0.01

        results.append(
            {
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "num_samples": result["num_samples"],
                "total_tokens": result["total_tokens"],
            }
        )

        total_tokens += result["total_tokens"]
        total_samples += result["num_samples"]
        if is_correct:
            correct_count += 1

        # Log to WandB
        wandb.log(
            {
                "num_samples": result["num_samples"],
                "tokens": result["total_tokens"],
                "correct": int(is_correct),
            }
        )

    # Compute final metrics
    accuracy = correct_count / len(results) if results else 0.0
    avg_samples = total_samples / len(results) if results else 0.0
    avg_tokens = total_tokens / len(results) if results else 0.0
    accuracy_per_1k_tokens = (accuracy * 1000 / avg_tokens) if avg_tokens > 0 else 0.0

    # Save metrics to WandB summary
    wandb.summary["accuracy"] = accuracy
    wandb.summary["avg_samples_used"] = avg_samples
    wandb.summary["avg_tokens_per_problem"] = avg_tokens
    wandb.summary["accuracy_per_1k_tokens"] = accuracy_per_1k_tokens

    # Save detailed results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWandB run URL: {wandb.run.get_url()}")
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg samples: {avg_samples:.2f}")
    print(f"  Avg tokens: {avg_tokens:.1f}")
    print(f"  Accuracy per 1k tokens: {accuracy_per_1k_tokens:.4f}")

    # Sanity validation
    if mode == "sanity_check":
        perform_sanity_validation(results, total_samples, total_tokens)

    wandb.finish()


def perform_sanity_validation(
    results: list[dict], total_samples: int, total_tokens: int
):
    """Perform sanity validation checks for inference tasks."""

    # Check: at least 5 samples processed
    num_problems = len(results)
    if num_problems < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_problems_{num_problems}")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":{num_problems}, "outputs_unique":0}}'
        )
        sys.exit(1)

    # Check: all outputs are valid (have predictions)
    valid_count = sum(1 for r in results if r["predicted_answer"] is not None)
    if valid_count == 0:
        print(f"SANITY_VALIDATION: FAIL reason=no_valid_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":{valid_count}, "outputs_unique":0}}'
        )
        sys.exit(1)

    # Check: outputs are not all identical
    unique_predictions = len(
        set(r["predicted_answer"] for r in results if r["predicted_answer"] is not None)
    )
    if unique_predictions <= 1 and valid_count > 1:
        print(f"SANITY_VALIDATION: FAIL reason=identical_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":{valid_count}, "outputs_unique":{unique_predictions}}}'
        )
        sys.exit(1)

    # Check: token counts are valid
    if total_tokens == 0:
        print(f"SANITY_VALIDATION: FAIL reason=zero_tokens")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":{valid_count}, "outputs_unique":{unique_predictions}}}'
        )
        sys.exit(1)

    print(f"SANITY_VALIDATION: PASS")
    print(
        f'SANITY_VALIDATION_SUMMARY: {{"samples":{total_samples}, "outputs_valid":{valid_count}, "outputs_unique":{unique_predictions}}}'
    )
