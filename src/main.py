"""Main orchestrator for running inference experiments."""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Orchestrate inference run.

    Args:
        cfg: Hydra configuration
    """
    # Print config for debugging
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Ensure results directory exists
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # This is an inference-only task - call inference.py directly
    from src.inference import run_inference

    try:
        run_inference(cfg)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
