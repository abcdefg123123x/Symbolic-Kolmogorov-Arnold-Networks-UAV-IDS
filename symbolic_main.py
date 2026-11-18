
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.dataset import load_tabular_dataset, TorchDataset
from src.models.registry import get_model
from src.symbolic import (
    extract_symbolic_equations,
    evaluate_symbolic_equations,
)


# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

MODEL_NAME = "KAN"

N_RUNS_PER_MAXSIZE = 10
BASE_SEED = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_run_seed(seed):
    """Set random seeds before each symbolic extraction run."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def update_yaml_bin_threshold(config_path, best_threshold):
    """
    Update bin_threshold in the YAML file while preserving its formatting.
    """

    with open(config_path, "r") as file:
        content = file.read()

    formatted_value = f"{best_threshold:.4f}".rstrip("0").rstrip(".")

    if "." not in formatted_value:
        formatted_value += ".0"

    if "optimal_values:" in content:

        if re.search(r"^\s*bin_threshold\s*:", content, flags=re.MULTILINE):
            content = re.sub(
                r"(^\s*bin_threshold\s*:\s*).*$",
                lambda match: match.group(1) + formatted_value,
                content,
                flags=re.MULTILINE,
            )

        else:
            content = re.sub(
                r"(optimal_values:\s*\n)",
                r"\1  bin_threshold: " + formatted_value + "\n",
                content,
                count=1,
            )

    else:
        content = (
            content.rstrip()
            + f"\n\noptimal_values:\n  bin_threshold: {formatted_value}\n"
        )

    with open(config_path, "w") as file:
        file.write(content)

    print(
        f"Updated {config_path} with bin_threshold: {formatted_value}"
    )


# ---------------------------------------------------------------------
# Binary classification plotting
# ---------------------------------------------------------------------

def plot_binary_threshold_sweep(
    thresholds,
    f1_history,
    best_threshold,
    dataset_name,
    checkpoint_dir,
):
    """Plot class-level F1 scores across different decision thresholds."""

    plt.figure(figsize=(9, 5))

    plt.plot(
        thresholds,
        f1_history[:, 0],
        label="Benign (Class 0) F1",
        linewidth=2,
    )

    plt.plot(
        thresholds,
        f1_history[:, 1],
        label="Malicious (Class 1) F1",
        linewidth=2,
    )

    plt.axvline(
        x=best_threshold,
        linestyle="--",
        label=f"Optimal threshold: {best_threshold:.2f}",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("F1-Score")
    plt.title(f"Binary Classification Threshold Sweep ({dataset_name})")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(
        checkpoint_dir,
        "threshold_vs_f1_classes.png",
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved threshold plot to {plot_path}")


# ---------------------------------------------------------------------
# Multiclass classification plotting
# ---------------------------------------------------------------------

def plot_multiclass_complexity_sweep(
    complexity_performance_tracking,
    best_maxsize,
    dataset_name,
    checkpoint_dir,
    encoder,
):
    """
    Plot mean validation macro F1 and standard deviation for each maxsize.
    Also plot the mean F1 score for each individual class.
    """

    if not complexity_performance_tracking:
        return

    maxsizes = sorted(complexity_performance_tracking.keys())

    mean_macro_f1 = [
        complexity_performance_tracking[ms]["mean_macro_f1"]
        for ms in maxsizes
    ]

    std_macro_f1 = [
        complexity_performance_tracking[ms]["std_macro_f1"]
        for ms in maxsizes
    ]

    # Mean macro F1 with standard deviation error bars.
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        maxsizes,
        mean_macro_f1,
        yerr=std_macro_f1,
        marker="o",
        capsize=5,
        linewidth=2,
        label="Mean Macro F1 ± 1 SD",
    )

    plt.axvline(
        x=best_maxsize,
        linestyle="--",
        alpha=0.7,
        label=f"Selected maxsize ({best_maxsize})",
    )

    plt.xlabel("Equation Maxsize")
    plt.ylabel("Validation Macro F1")
    plt.title(
        f"Symbolic Complexity vs Validation Performance ({dataset_name})"
    )
    plt.xticks(maxsizes)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        checkpoint_dir,
        "complexity_vs_macro_f1.png",
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot mean F1 for each class.
    class_ids = sorted(
        next(iter(complexity_performance_tracking.values()))[
            "class_f1_mean"
        ].keys()
    )

    plt.figure(figsize=(10, 6))

    for class_id in class_ids:

        class_scores = [
            complexity_performance_tracking[ms][
                "class_f1_mean"
            ][class_id]
            for ms in maxsizes
        ]

        class_name = (
            encoder.classes_[class_id]
            if hasattr(encoder, "classes_")
            else f"Class {class_id}"
        )

        plt.plot(
            maxsizes,
            class_scores,
            marker="o",
            linestyle="-",
            label=f"{class_name} F1",
        )

    plt.axvline(
        x=best_maxsize,
        linestyle="--",
        alpha=0.7,
        label=f"Selected maxsize ({best_maxsize})",
    )

    plt.xlabel("Equation Maxsize")
    plt.ylabel("Mean Validation F1")
    plt.title(
        f"Mean Class F1 vs Symbolic Complexity ({dataset_name})"
    )
    plt.xticks(maxsizes)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(
        checkpoint_dir,
        "complexity_vs_class_f1.png",
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved complexity plots to {checkpoint_dir}")


# ---------------------------------------------------------------------
# Repeated maxsize evaluation
# ---------------------------------------------------------------------

def run_maxsize_sweep(
    maxsize_list,
    n_runs,
    model,
    model_name,
    train_loader,
    X_val,
    y_val,
    config,
    device,
    num_classes,
):
    """
    Run fresh symbolic extractions for every maxsize.

    Each candidate maxsize is evaluated n_runs times on the same
    validation set. Mean and sample standard deviation are calculated
    for macro F1 and individual class F1 scores.

    The best maxsize is selected using mean validation macro F1.
    """

    results = {}

    y_val_flat = np.asarray(y_val).flatten()
    class_labels = np.arange(num_classes)

    for maxsize in maxsize_list:

        print("\n" + "=" * 55)
        print(f"Evaluating maxsize = {maxsize}")
        print(f"Number of runs = {n_runs}")
        print("=" * 55)

        run_macro_f1 = []
        run_class_f1 = []

        for run in range(n_runs):

            seed = BASE_SEED + run
            set_run_seed(seed)

            print(
                f"\nMaxsize {maxsize} | "
                f"Run {run + 1}/{n_runs} | Seed {seed}"
            )

            # Perform a fresh symbolic extraction.
            artifacts = extract_symbolic_equations(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                device=device,
                config=config,
                maxsize=maxsize,
            )

            # Evaluate the extracted equations on the validation set.
            eval_metrics = evaluate_symbolic_equations(
                symbolic_results=artifacts,
                X_test=X_val,
                y_test=y_val,
                model_name=model_name,
                config=config,
                threshold=0.5,
            )

            predictions = np.asarray(
                eval_metrics["predictions"]
            ).flatten()

            if len(predictions) != len(y_val_flat):
                raise ValueError(
                    f"Prediction count mismatch for maxsize {maxsize}, "
                    f"run {run + 1}: expected {len(y_val_flat)}, "
                    f"got {len(predictions)}."
                )

            # Calculate macro F1 using all class labels.
            macro_f1 = f1_score(
                y_val_flat,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )

            # Calculate F1 separately for every class.
            class_f1 = f1_score(
                y_val_flat,
                predictions,
                labels=class_labels,
                average=None,
                zero_division=0,
            )

            run_macro_f1.append(float(macro_f1))
            run_class_f1.append(class_f1.tolist())

            print(f"Run Macro F1: {macro_f1:.4f}")

        # Calculate summary statistics across the repeated runs.
        run_macro_f1 = np.asarray(run_macro_f1)
        run_class_f1 = np.asarray(run_class_f1)

        results[maxsize] = {
            "run_macro_f1": run_macro_f1.tolist(),

            "mean_macro_f1": float(
                np.mean(run_macro_f1)
            ),

            "std_macro_f1": float(
                np.std(run_macro_f1, ddof=1)
            ) if n_runs > 1 else 0.0,

            "class_f1_mean": {
                int(class_id): float(
                    np.mean(run_class_f1[:, class_id])
                )
                for class_id in class_labels
            },

            "class_f1_std": {
                int(class_id): float(
                    np.std(
                        run_class_f1[:, class_id],
                        ddof=1,
                    )
                ) if n_runs > 1 else 0.0
                for class_id in class_labels
            },
        }

        summary = results[maxsize]

        print(f"\nSummary for maxsize = {maxsize}")
        print(
            f"Mean Macro F1: {summary['mean_macro_f1']:.4f} "
            f"+/- {summary['std_macro_f1']:.4f}"
        )

    # Select the maxsize with the highest mean validation macro F1.
    best_maxsize = max(
        results,
        key=lambda ms: results[ms]["mean_macro_f1"],
    )

    best_results = results[best_maxsize]

    print("\n" + "=" * 55)
    print("MAXSIZE SWEEP SUMMARY")
    print("=" * 55)

    for maxsize in sorted(results):

        summary = results[maxsize]

        print(
            f"Maxsize {maxsize:>3}: "
            f"{summary['mean_macro_f1']:.4f} "
            f"+/- {summary['std_macro_f1']:.4f}"
        )

    print("-" * 55)

    print(
        f"Selected maxsize: {best_maxsize}\n"
        f"Mean validation Macro F1: "
        f"{best_results['mean_macro_f1']:.4f} "
        f"+/- {best_results['std_macro_f1']:.4f}"
    )

    print("=" * 55)

    return results, best_maxsize


# ---------------------------------------------------------------------
# Main symbolic extraction pipeline
# ---------------------------------------------------------------------

def main(config_path, maxsize_list, threshold_candidates):

    # Load experiment configuration.
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print(f"Using device: {DEVICE}")

    dataset_name = config["data"]["name"]
    binary_mode = config["data"]["binary_mode"]

    classification_mode = (
        "binary" if binary_mode == 1 else "multi"
    )

    checkpoint_dir = os.path.join(
        "saved_checkpoints",
        dataset_name,
        classification_mode,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    hyperparam_path = os.path.join(
        "optimal",
        dataset_name,
        classification_mode,
        f"best_{MODEL_NAME}.json",
    )

    if not os.path.exists(hyperparam_path):
        print(
            f"Hyperparameter file not found for {MODEL_NAME}: "
            f"{hyperparam_path}"
        )
        return

    print("\n" + "=" * 55)
    print(
        f"Loading saved model and running symbolic pipeline: "
        f"{MODEL_NAME} ({classification_mode.upper()})"
    )
    print("=" * 55)

    with open(hyperparam_path, "r") as file:
        best_params = json.load(file)

    # Load the training, validation and test splits.
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

    input_dim = X_train.shape[-1]

    # Include both training and validation labels when determining
    # the total number of classes.
    y_train_final = np.concatenate([y_train, y_val])
    X_train_final = np.concatenate([X_train, X_val], axis=0)

    num_classes = len(np.unique(y_train_final))
    batch_size = best_params["batch_size"]

    # Load the trained KAN model.
    model = get_model(
        MODEL_NAME,
        best_params,
        input_dim,
        num_classes,
    )

    torch_model_path = os.path.join(
        checkpoint_dir,
        f"{MODEL_NAME}_weights.pth",
    )

    if not os.path.exists(torch_model_path):
        print(
            f"Saved weights not found at {torch_model_path}. "
            "Train the model first."
        )
        return

    model.load_state_dict(
        torch.load(torch_model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    print(f"Successfully loaded model weights from {torch_model_path}")

    # -----------------------------------------------------------------
    # Binary classification
    # -----------------------------------------------------------------

    if binary_mode == 1:

        print(
            "\n--- Binary Pipeline: "
            "Extracting Equations and Tuning Threshold ---"
        )

        train_loader_tuning = DataLoader(
            TorchDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,
        )

        # Extract symbolic equations using the training split.
        symbolic_artifacts = extract_symbolic_equations(
            model=model,
            model_name=MODEL_NAME,
            train_loader=train_loader_tuning,
            device=DEVICE,
            config=config,
        )

        pysr_model = symbolic_artifacts["model"]

        # Generate validation predictions.
        raw_preds = pysr_model.predict(X_val)

        # Convert symbolic outputs to probabilities.
        raw_preds = np.clip(raw_preds, -500, 500)
        preds_prob = 1 / (1 + np.exp(-raw_preds))

        f1_history = []
        best_threshold = 0.5
        best_val_f1 = -1.0

        y_val_flat = np.asarray(y_val).flatten()

        # Find the threshold with the highest validation macro F1.
        for threshold in threshold_candidates:

            binary_preds = (
                preds_prob >= threshold
            ).astype(int)

            scores = f1_score(
                y_val_flat,
                binary_preds,
                labels=[0, 1],
                average=None,
                zero_division=0,
            )

            f1_history.append(scores)

            macro_f1 = np.mean(scores)

            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                best_threshold = threshold

        f1_history = np.asarray(f1_history)

        print(
            f"\nOptimal binary threshold: {best_threshold:.2f}"
            f"\nValidation Macro F1: {best_val_f1:.4f}"
        )

        # Store the selected threshold in the configuration file.
        update_yaml_bin_threshold(
            config_path,
            best_threshold,
        )

        # Plot threshold performance.
        plot_binary_threshold_sweep(
            threshold_candidates,
            f1_history,
            best_threshold,
            dataset_name,
            checkpoint_dir,
        )

        # Re-extract symbolic equations using training and validation data.
        train_loader_final = DataLoader(
            TorchDataset(X_train_final, y_train_final),
            batch_size=batch_size,
            shuffle=False,
        )

        final_artifacts = extract_symbolic_equations(
            model=model,
            model_name=MODEL_NAME,
            train_loader=train_loader_final,
            device=DEVICE,
            config=config,
        )

        print("\n--- Final Test Evaluation (Binary) ---")

        eval_metrics = evaluate_symbolic_equations(
            symbolic_results=final_artifacts,
            X_test=X_test,
            y_test=y_test,
            model_name=MODEL_NAME,
            config=config,
            threshold=best_threshold,
        )

    # -----------------------------------------------------------------
    # Multiclass classification
    # -----------------------------------------------------------------

    else:

        print(
            "\n--- Multiclass Pipeline: "
            "Repeated Maxsize Evaluation ---"
        )

        train_loader_tuning = DataLoader(
            TorchDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,
        )

        # Perform 10 fresh symbolic extractions for each candidate maxsize.
        complexity_performance_tracking, best_maxsize = (
            run_maxsize_sweep(
                maxsize_list=maxsize_list,
                n_runs=N_RUNS_PER_MAXSIZE,
                model=model,
                model_name=MODEL_NAME,
                train_loader=train_loader_tuning,
                X_val=X_val,
                y_val=y_val,
                config=config,
                device=DEVICE,
                num_classes=num_classes,
            )
        )

        best_results = complexity_performance_tracking[best_maxsize]

        print(
            f"\nSelected maxsize: {best_maxsize}"
            f"\nValidation Macro F1: "
            f"{best_results['mean_macro_f1']:.4f} "
            f"+/- {best_results['std_macro_f1']:.4f}"
        )

        # Save the individual run scores and summary statistics.
        results_path = os.path.join(
            checkpoint_dir,
            "maxsize_sweep_results.json",
        )

        with open(results_path, "w") as file:
            json.dump(
                {
                    str(maxsize): results
                    for maxsize, results
                    in complexity_performance_tracking.items()
                },
                file,
                indent=4,
            )

        print(f"Saved maxsize results to {results_path}")

        # Plot the mean macro F1, standard deviation and class-level F1.
        plot_multiclass_complexity_sweep(
            complexity_performance_tracking,
            best_maxsize,
            dataset_name,
            checkpoint_dir,
            encoder,
        )

        # Re-extract the final equations using the selected maxsize.
        train_loader_final = DataLoader(
            TorchDataset(X_train_final, y_train_final),
            batch_size=batch_size,
            shuffle=False,
        )

        final_artifacts = extract_symbolic_equations(
            model=model,
            model_name=MODEL_NAME,
            train_loader=train_loader_final,
            device=DEVICE,
            config=config,
            maxsize=best_maxsize,
        )

        print("\n--- Final Test Evaluation (Multiclass) ---")

        eval_metrics = evaluate_symbolic_equations(
            symbolic_results=final_artifacts,
            X_test=X_test,
            y_test=y_test,
            model_name=MODEL_NAME,
            config=config,
        )

    # -----------------------------------------------------------------
    # Final evaluation results
    # -----------------------------------------------------------------

    print("\nConfusion Matrix:\n")
    print(eval_metrics["conf_mat"])

    print("\nClassification Report:\n")
    print(eval_metrics["clasif_rep"])

    # Save the final symbolic artifacts.
    symbolic_path = os.path.join(
        checkpoint_dir,
        f"{MODEL_NAME}_symbolic_artifacts.joblib",
    )

    joblib.dump(final_artifacts, symbolic_path)

    print(
        f"\nSuccessfully saved final symbolic artifacts to "
        f"{symbolic_path}"
    )


# ---------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------

if __name__ == "__main__":

    config_files = [
        "configs/d1.yaml",
    ]

    custom_maxsize_list = [
        10,
        15,
        20,
        25,
        30,
        40,
        50,
    ]

    custom_threshold_list = np.arange(
        0.1,
        0.91,
        0.01,
    )

    for config_path in config_files:

        main(
            config_path,
            maxsize_list=custom_maxsize_list,
            threshold_candidates=custom_threshold_list,
        )
