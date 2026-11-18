def get_search_space(trial, model_name):
    """
    Returns a dictionary of hyperparameters for Optuna to explore.
    """
    if model_name =="KAN":
        return {
            # Architectural parameters
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64]),
            "kan_hidden": trial.suggest_int("kan_hidden", 16, 64),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),

            # KAN spline parameters
            "grid_size": trial.suggest_int("grid_size", 3, 5),
            "spline_order": 3,

            # Optimisation and training configuration
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256])
            #"gamma": trial.suggest_float("gamma", 0.4, 4.0)
        }

    elif model_name == "RandomForest":
        return {
            # Tree ensemble structural and optimisation parameters
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
        }

    elif model_name == "MLP":
        # Changed minimum from 1 to 2
        num_layers = trial.suggest_int("num_layers", 2, 4)

        space = {
            "num_layers": num_layers,
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
        }

        # Dynamically add the layer sizes based on the chosen num_layers
        for i in range(1, num_layers + 1):
            space[f"layer_{i}_size"] = trial.suggest_int(f"layer_{i}_size", 64, 512)

        return space

    elif model_name == "CNN":
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)

        space = {
            "num_conv_layers": num_conv_layers,
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size",
                                                    [64, 128, 256])
        }

        for i in range(1, num_conv_layers + 1):
            space[f"layer_{i}_channels"] = trial.suggest_categorical(
                f"layer_{i}_channels", [32, 64, 128, 256])

        return space

    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")
