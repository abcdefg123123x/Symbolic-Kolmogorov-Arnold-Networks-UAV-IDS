"""
main.py

Creating and saving ML models. Not symbolic for Symbolic KAN.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import yaml
import torch
import json
import os
import random
import joblib

from torch.utils.data import DataLoader

from src.dataset import load_tabular_dataset, TorchDataset
from src.models.registry import get_model
from src.training.kan_train import train_kan_final, evaluate_kan_model
from src.training.rf_train import train_rf_final, evaluate_rf_model
from src.training.mlp_train import train_mlp_final, evaluate_mlp_model
from src.training.lightgbm_train import train_lgb_final, evaluate_lgb_model
from src.training.cnn_1d_train import train_cnn_final, evaluate_cnn_model
from src.utils import evaluate_model


def set_seed(seed):
    """Locks all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path, models):
    set_seed(1107)

    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_name = config["data"]["name"]
    binary_mode = config["data"]["binary_mode"]
    clasif_mode = "binary" if binary_mode == 1 else "multi"

    checkpoint_dir = f"saved_checkpoints/{dataset_name}/{clasif_mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for m in models:
        hyperparam_path = f"optimal/{dataset_name}/{clasif_mode}/best_{m}.json"

        if not os.path.exists(hyperparam_path):
            print(f"Hyperparameter file not found for {m} at {hyperparam_path}. Skipping...")
            continue

        print(f"\n==========================================")
        print(f"Processing Model: {m}")
        print(f"==========================================")

        with open(hyperparam_path, "r") as f:
            best_params = json.load(f)
        print(f"Loaded optimal parameters for {m}: {best_params}")

        is_torch_model = m in ["MLP", "KAN", "CNN"]

        # 2. Re-load tabular dataset exclusively
        X_train, X_val, X_test, y_train, y_val, y_test, encoder, scaler = load_tabular_dataset(config)

        X_train_final = np.concatenate([X_train, X_val], axis=0)
        y_train_final = np.concatenate([y_train, y_val], axis=0)

        print(f"Final training data shape: {X_train_final.shape}")
        print(f"Held-out test data shape:  {X_test.shape}")

        input_dim = X_train_final.shape[-1]
        num_classes = len(np.unique(y_train_final))

        print(f"Model setup: Model={m}, InputDim={input_dim}, Classes={num_classes}")

        model = get_model(m, best_params, input_dim, num_classes)

        if is_torch_model:
            batch_size = best_params["batch_size"]
            train_loader = DataLoader(TorchDataset(X_train_final, y_train_final), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TorchDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        symbolic_artifacts = None

        if m == "KAN":
            model = train_kan_final(model, train_loader, device, config, y_train_final, lr=best_params["lr"]) # gamma=best_params["gamma"])
            final_metrics = evaluate_kan_model(model, test_loader, device)

        elif m == "RandomForest":
            model = train_rf_final(model, X_train_final, y_train_final)
            final_metrics = evaluate_rf_model(model, X_test, y_test)

        elif m == "LightGBM":
            model = train_lgb_final(model, X_train_final, y_train_final)
            final_metrics = evaluate_lgb_model(model, X_test, y_test)

        elif m == "MLP":
            model = train_mlp_final(model, train_loader, device, config, lr=best_params["lr"])
            final_metrics = evaluate_mlp_model(model, test_loader, device)

        elif m == "CNN":
            model = train_cnn_final(model, train_loader, device, config, lr=best_params["lr"])
            final_metrics = evaluate_cnn_model(model, test_loader, device)

        # Save model checkpoints
        if is_torch_model:
            torch_model_path = os.path.join(checkpoint_dir, f"{m}_weights.pth")
            torch.save(model.state_dict(), torch_model_path)
            print(f"Saved PyTorch model {m} weights to {torch_model_path}")

        elif m in ["RandomForest", "LightGBM"]:
            model_path = os.path.join(checkpoint_dir, f"{m}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {m} model to: {model_path}")

        # Can ignore for now in main.py (main.py for creating and saving models themselves, not symbolic yet)
        if symbolic_artifacts is not None:
            symbolic_path = os.path.join(checkpoint_dir, f"{m}_symbolic_artifacts.joblib")
            joblib.dump(symbolic_artifacts, symbolic_path)
            print(f"Saved PySR symbolic equations to: {symbolic_path}")

        acc = evaluate_model(final_metrics["predictions"], y_test, encoder, config, final_metrics["probabilities"], m)

        print("Final macro F1:", final_metrics['macro_f1'])
        print("Accuracy:", final_metrics['accuracy'])


if __name__ == "__main__":
    config_files = [
        "configs/d1.yaml"
    ]

    models = ["LightGBM", "RandomForest"]

    for config_path in config_files:
        main(config_path, models)
