
import json
import os
import threading
import time

import joblib
import numpy as np
import psutil
import torch
import yaml

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import load_tabular_dataset, TorchDataset
from src.models.registry import get_model
from src.symbolic import evaluate_symbolic_equations
from src.training.kan_train import evaluate_kan_model
from src.training.rf_train import evaluate_rf_model
from src.training.mlp_train import evaluate_mlp_model
from src.training.lightgbm_train import evaluate_lgb_model
from src.training.cnn_1d_train import evaluate_cnn_model


# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

DEVICE = torch.device("cpu")

# Keep CPU usage consistent across model evaluations.
TORCH_NUM_THREADS = 2
TORCH_NUM_INTEROP_THREADS = 1
CPU_LIMIT = 2

# Sample process memory every millisecond.
MEMORY_SAMPLE_INTERVAL = 0.001

# A batch size of one measures single-sample inference latency.
TEST_BATCH_SIZE = 1

# Standard models included in the evaluation.
MODELS = [
    "KAN",
    "RandomForest",
    "LightGBM",
    "MLP",
    "CNN",
]

# SymbolicKAN is evaluated separately because it uses saved equations
# rather than a conventional trained model checkpoint.
SYMBOLIC_MODELS = ["SymbolicKAN"]


def configure_environment():
    """Set CPU and threading limits for the experiment."""
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

    process = psutil.Process(os.getpid())

    # Use two CPUs from the CPUs available to this process.
    # This also works when the process is restricted by a container.
    available_cpus = process.cpu_affinity()
    process.cpu_affinity(available_cpus[:CPU_LIMIT])


# ---------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------

class PeakMemoryTracker:
    """
    Track the Python process's resident memory during an operation.

    Memory is reported in MB. The tracker records the RSS immediately
    before measurement and continuously samples it in a background
    thread until stop() is called.
    """

    def __init__(self, interval=MEMORY_SAMPLE_INTERVAL):
        self.interval = interval
        self.process = psutil.Process(os.getpid())

        self.baseline_mem = 0.0
        self.peak_mem = 0.0

        self._stop_event = threading.Event()
        self._thread = None

    def _get_rss_mb(self):
        """Return the current process RSS in megabytes."""
        return self.process.memory_info().rss / (1024 ** 2)

    def _poll_memory(self):
        """Sample RSS until the tracker is stopped."""
        while not self._stop_event.is_set():
            try:
                current_mem = self._get_rss_mb()
                self.peak_mem = max(self.peak_mem, current_mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            self._stop_event.wait(self.interval)

    def start(self):
        """Record the baseline and start background memory sampling."""
        self.baseline_mem = self._get_rss_mb()
        self.peak_mem = self.baseline_mem

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._poll_memory,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """
        Stop sampling and return baseline and peak RSS.

        A final RSS reading is taken after the sampling thread stops.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        try:
            final_mem = self._get_rss_mb()
            self.peak_mem = max(self.peak_mem, final_mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return self.baseline_mem, self.peak_mem


# ---------------------------------------------------------------------
# Performance reporting
# ---------------------------------------------------------------------

def compute_and_print_metrics(
    model_name,
    metrics,
    total_latency,
    baseline_mem,
    peak_mem,
    num_classes,
    y_test,
):
    """Calculate ROC-AUC and print a consistent evaluation report."""

    predictions = metrics["predictions"]
    probabilities = metrics.get("probabilities")

    # Calculate ROC-AUC when probability predictions are available.
    roc_auc_str = "N/A"

    if probabilities is not None:
        try:
            probabilities = np.asarray(probabilities)

            if num_classes == 2:
                # For binary classification, use the positive-class score.
                if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                    y_score = probabilities[:, 1]
                else:
                    y_score = probabilities

                roc_auc = roc_auc_score(y_test, y_score)

            else:
                roc_auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )

            roc_auc_str = f"{roc_auc:.4f}"

        except (ValueError, IndexError) as error:
            roc_auc_str = f"Unavailable ({error})"

    total_samples = len(predictions)

    per_sample_latency = (
        total_latency / total_samples if total_samples > 0 else 0
    )

    # Report both the total process memory and the increase during
    # the measured evaluation period.
    additional_peak_mem = max(0.0, peak_mem - baseline_mem)

    print(f"\n--- PERFORMANCE REPORT: {model_name} ---")
    print(f"Total test samples          : {total_samples}")
    print(f"Accuracy                    : {metrics['accuracy']:.4f}")
    print(f"Macro F1                    : {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC (Macro OvR)         : {roc_auc_str}")
    print(f"Total inference time        : {total_latency:.6f} seconds")
    print(f"Per-sample latency          : {per_sample_latency * 1000:.4f} ms")
    print(f"Baseline process RAM        : {baseline_mem:.2f} MB")
    print(f"Peak process RAM            : {peak_mem:.2f} MB")
    print(f"Additional peak RAM         : {additional_peak_mem:.2f} MB")
    print("============================================")


# ---------------------------------------------------------------------
# Dataset and checkpoint paths
# ---------------------------------------------------------------------

def get_experiment_paths(config):
    """Build the checkpoint and hyperparameter paths for this dataset."""

    dataset_name = config["data"]["name"]
    classification_mode = (
        "binary" if config["data"]["binary_mode"] == 1 else "multi"
    )

    checkpoint_dir = os.path.join(
        "saved_checkpoints",
        dataset_name,
        classification_mode,
    )

    hyperparameter_dir = os.path.join(
        "optimal",
        dataset_name,
        classification_mode,
    )

    return dataset_name, checkpoint_dir, hyperparameter_dir


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

def evaluate_symbolic_model(
    model_name,
    checkpoint_dir,
    hyperparameter_dir,
    X_test,
    y_test,
    config,
    num_classes,
):
    """Evaluate the saved symbolic equations extracted from the KAN."""

    symbolic_path = os.path.join(
        checkpoint_dir,
        "KAN_symbolic_artifacts.joblib",
    )
    hyperparameter_path = os.path.join(
        hyperparameter_dir,
        "best_KAN.json",
    )

    if not os.path.exists(symbolic_path) or not os.path.exists(hyperparameter_path):
        print("Symbolic artifacts or KAN hyperparameters not found. Skipping.")
        return

    symbolic_artifacts = joblib.load(symbolic_path)

    tracker = PeakMemoryTracker()
    tracker.start()
    start_time = time.perf_counter()

    metrics = evaluate_symbolic_equations(
        symbolic_results=symbolic_artifacts,
        X_test=X_test,
        y_test=y_test,
        model_name="KAN",
        config=config,
        threshold=(
            config["optimal_values"]["bin_threshold"]
            if config["data"]["binary_mode"] == 1
            else None
        ),
    )

    total_latency = time.perf_counter() - start_time
    baseline_mem, peak_mem = tracker.stop()

    compute_and_print_metrics(
        model_name,
        metrics,
        total_latency,
        baseline_mem,
        peak_mem,
        num_classes,
        y_test,
    )


def evaluate_standard_model(
    model_name,
    checkpoint_dir,
    hyperparameter_dir,
    X_test,
    y_test,
    input_dim,
    num_classes,
):
    """Load a trained model and evaluate it on the test set."""

    hyperparameter_path = os.path.join(
        hyperparameter_dir,
        f"best_{model_name}.json",
    )

    if not os.path.exists(hyperparameter_path):
        print(f"Hyperparameter file not found: {hyperparameter_path}")
        return

    with open(hyperparameter_path, "r") as file:
        best_params = json.load(file)

    model = get_model(
        model_name,
        best_params,
        input_dim,
        num_classes,
    )

    is_torch_model = model_name in ["KAN", "MLP", "CNN"]

    # Load the appropriate model checkpoint.
    if is_torch_model:
        model_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_weights.pth",
        )

        if not os.path.exists(model_path):
            print(f"Model weights not found: {model_path}")
            return

        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()

        test_loader = DataLoader(
            TorchDataset(X_test, y_test),
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
        )

    else:
        model_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_model.joblib",
        )

        if not os.path.exists(model_path):
            print(f"Saved model not found: {model_path}")
            return

        model = joblib.load(model_path)
        test_loader = None

    # Measure inference time and process RAM after the model is loaded.
    tracker = PeakMemoryTracker()
    tracker.start()
    start_time = time.perf_counter()

    if model_name == "KAN":
        metrics = evaluate_kan_model(model, test_loader, DEVICE)

    elif model_name == "MLP":
        metrics = evaluate_mlp_model(model, test_loader, DEVICE)

    elif model_name == "CNN":
        metrics = evaluate_cnn_model(model, test_loader, DEVICE)

    elif model_name == "RandomForest":
        metrics = evaluate_rf_model(model, X_test, y_test)

    elif model_name == "LightGBM":
        metrics = evaluate_lgb_model(model, X_test, y_test)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    total_latency = time.perf_counter() - start_time
    baseline_mem, peak_mem = tracker.stop()

    compute_and_print_metrics(
        model_name,
        metrics,
        total_latency,
        baseline_mem,
        peak_mem,
        num_classes,
        y_test,
    )


# ---------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------

def main(config_path, models):
    """Load the dataset and evaluate each requested model."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_name, checkpoint_dir, hyperparameter_dir = (
        get_experiment_paths(config)
    )

    print("\n============================================")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {DEVICE}")
    print("============================================")

    # Load the dataset once and reuse the test split for every model.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        encoder,
        scaler,
    ) = load_tabular_dataset(config)

    # Combine training and validation labels to determine the number
    # of classes used by the final models.
    y_train_final = np.concatenate([y_train, y_val])
    num_classes = len(np.unique(y_train_final))
    input_dim = X_train.shape[-1]

    for model_name in models:
        print(f"\nEvaluating model: {model_name}")
        print("--------------------------------------------")

        if model_name == "SymbolicKAN":
            evaluate_symbolic_model(
                model_name,
                checkpoint_dir,
                hyperparameter_dir,
                X_test,
                y_test,
                config,
                num_classes,
            )
        else:
            evaluate_standard_model(
                model_name,
                checkpoint_dir,
                hyperparameter_dir,
                X_test,
                y_test,
                input_dim,
                num_classes,
            )


# ---------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------

if __name__ == "__main__":
    configure_environment()

    config_files = [
        "configs/d1.yaml",
    ]

    # Evaluate all five standard models.
    models = MODELS

    # To include the symbolic KAN as well, use:
    # models = MODELS + SYMBOLIC_MODELS

    for config_path in config_files:
        main(config_path, models)
