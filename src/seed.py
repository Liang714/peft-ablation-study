"""Single source of truth for reproducibility. Every script (data prep,
training, evaluation) must call set_seed() exactly once at startup with the
seed from its ExperimentConfig — never hardcode a seed elsewhere."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set the seed for python, numpy, torch (CPU + CUDA), and transformers.

    Torch/transformers are imported lazily inside this function so that
    modules which only need deterministic Python/NumPy behavior (e.g. the
    data pipeline) don't require torch to be installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed — skipping torch seeding")

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        logger.debug("transformers not installed — skipping transformers seeding")

    logger.info("Seed set to %d", seed)
