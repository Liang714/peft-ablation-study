"""Thin MLflow wrapper — every run (baseline + every fine-tuning run) logs
here under a consistent experiment name / run naming scheme, per the project
brief's engineering standards."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def log_run(
    cfg: ExperimentConfig,
    run_name: str,
    metrics: dict[str, float],
    params: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Log one MLflow run. run_name is passed explicitly (rather than always
    built from cfg.logging.run_name_template) since non-training runs like
    the Step 3 baseline don't have a training `method` in the usual sense."""
    import mlflow

    mlflow.set_tracking_uri(cfg.logging.tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("model_name", cfg.model.name)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

    logger.info("Logged MLflow run '%s': metrics=%s", run_name, metrics)
