
import json
import os
import random

import matplotlib
matplotlib.use("Agg")

import numpy as np
import optuna
import torch
import yaml

from src.tuning.objective import Objective
from src.utils import save_optuna_plots


# Keep the tuning runs reproducible.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path, models):
    set_seed(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset and model configuration.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["data"]["name"]

    print("\n========================================")
    print(f"Loading configuration from: {dataset_name}")
    print(f"Tuning on device: {device}")
    print("========================================")

    # Run a separate Optuna study for each model.
    for model_name in models:
        print(f"\n--- Starting optimisation for: {model_name} ---")

        study = optuna.create_study(direction="maximize")

        objective = Objective(
            model_name=model_name,
            config=config,
            device=device,
        )

        study.optimize(objective, n_trials=5)

        # Save the best hyperparameters for this model.
        classification_mode = (
            "binary" if config["data"]["binary_mode"] == 1 else "multi"
        )

        output_dir = os.path.join(
            "optimal",
            dataset_name,
            classification_mode,
        )
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"best_{model_name}.json",
        )

        with open(output_path, "w") as f:
            json.dump(study.best_params, f, indent=4)

        print(f"Finished {model_name}. Best parameters saved to {output_path}")
        print(f"Best score: {study.best_value:.4f}")

        # Generate and save the plots for this study.
        save_optuna_plots(
            study,
            model_name=model_name,
            config=config,
        )


if __name__ == "__main__":
    # Add the configuration files you want to run here.
    config_files = [
        # "configs/dataset1.yaml",
    ]

    # Models to include in the tuning process.
    models = ["KAN", "RandomForest", "LightGBM", "MLP", "CNN"]

    # Run the tuning process for each configuration.
    for config_path in config_files:
        main(config_path, models)
