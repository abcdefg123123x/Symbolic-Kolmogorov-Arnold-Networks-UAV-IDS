
"""
objective.py

Defines the Optuna objective used for hyperparameter tuning.

The Objective class handles data loading, model initialisation and
training for the selected model. It supports both traditional
machine learning models and PyTorch-based models.

The validation Macro F1-score is returned to Optuna so that
different hyperparameter configurations can be compared.
"""

import numpy as np

from torch.utils.data import DataLoader

from src.dataset import load_tabular_dataset, TorchDataset
from src.tuning.search_spaces import get_search_space
from src.models.registry import get_model

from src.training.kan_train import train_kan_optuna
from src.training.rf_train import train_rf_optuna
from src.training.mlp_train import train_mlp_optuna
from src.training.lightgbm_train import train_lgb_optuna
from src.training.cnn_1d_train import train_cnn_optuna


# ============================================================
# Optuna objective
# ============================================================

class Objective:
    """
    Handles the training and evaluation process for Optuna trials.

    Each trial samples a new set of hyperparameters, initialises
    the selected model and evaluates its performance on the
    validation dataset.

    The resulting validation Macro F1-score is used by Optuna
    to identify the best hyperparameter configuration.
    """

    def __init__(self, model_name, config, device):
        """
        Sets up the objective and loads the dataset.

        Args:
            model_name: Name of the model being tuned.
            config: Project configuration dictionary.
            device: Device used for PyTorch model training.
        """

        self.model_name = model_name
        self.config = config
        self.device = device

        # Identify models that require PyTorch DataLoaders.
        self.is_torch_model = model_name in ["MLP", "KAN", "CNN"]

        # Load and preprocess the dataset once before starting
        # the optimisation process.
        self.data = load_tabular_dataset(config)

        # Get the number of input features and output classes.
        self.input_size = self.data[0].shape[-1]
        self.y_train_raw = self.data[3]

        self.num_classes = len(np.unique(self.y_train_raw))

        print(
            f"Objective setup: Model={model_name}, "
            f"Input features={self.input_size}, "
            f"Classes={self.num_classes}"
        )

    # --------------------------------------------------------
    # Run an individual Optuna trial
    # --------------------------------------------------------

    def __call__(self, trial):
        """
        Trains and evaluates a model using the parameters sampled
        for the current Optuna trial.

        Args:
            trial: Current Optuna trial.

        Returns:
            Validation Macro F1-score.
        """

        # Sample a hyperparameter configuration for this model.
        params = get_search_space(trial, self.model_name)

        # Retrieve the training and validation data.
        # The test set is intentionally not used during tuning.
        (
            X_tr,
            X_v,
            _,
            y_tr,
            y_v,
            _,
            _,
            _
        ) = self.data

        # Create a fresh model for the current trial.
        model = get_model(
            self.model_name,
            params,
            self.input_size,
            self.num_classes
        )

        # ----------------------------------------------------
        # Traditional machine learning models
        # ----------------------------------------------------

        if self.model_name == "RandomForest":

            # Train the Random Forest and evaluate it on validation data.
            _, best_f1 = train_rf_optuna(
                model,
                X_tr,
                y_tr,
                X_v,
                y_v
            )

        elif self.model_name == "LightGBM":

            # LightGBM handles the tabular arrays directly.
            _, best_f1 = train_lgb_optuna(
                model,
                X_tr,
                y_tr,
                X_v,
                y_v
            )

        # ----------------------------------------------------
        # PyTorch models
        # ----------------------------------------------------

        elif self.is_torch_model:

            batch_size = params["batch_size"]

            # Convert the NumPy arrays into PyTorch datasets
            # before creating the training and validation loaders.
            train_dataset = TorchDataset(X_tr, y_tr)
            val_dataset = TorchDataset(X_v, y_v)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                persistent_workers=False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                persistent_workers=False
            )

            # Pass the sampled learning rate and other required
            # settings to the appropriate model training function.
            if self.model_name == "MLP":

                _, best_f1 = train_mlp_optuna(
                    model,
                    train_loader,
                    val_loader,
                    self.device,
                    self.config,
                    lr=params["lr"]
                )

            elif self.model_name == "KAN":

                _, best_f1 = train_kan_optuna(
                    model,
                    train_loader,
                    val_loader,
                    self.device,
                    self.config,
                    y_tr,
                    lr=params["lr"]
                )

            elif self.model_name == "CNN":

                _, best_f1 = train_cnn_optuna(
                    model,
                    train_loader,
                    val_loader,
                    self.device,
                    self.config,
                    lr=params["lr"]
                )

        else:
            # Prevent unsupported model names from silently
            # proceeding without a valid training routine.
            raise ValueError(
                f"Unsupported model name: {self.model_name}"
            )

        # Optuna uses this validation score to compare trials.
        return best_f1
