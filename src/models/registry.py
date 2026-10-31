# models/registry.py
from src.models.kan_model import KANModel
from src.models.rand_forest_model import build_rf_model
from src.models.mlp_model import build_mlp_model
from src.models.lightgbm_model import build_lightgbm_model
from src.models.cnn_1d_model import build_cnn_model

def get_model(model_name, params, input_size, num_classes):
    """
    Returns the initialised model class based on the name.
    """
    if model_name == "KAN":
        return KANModel(
            input_size=input_size,
            num_classes=num_classes,
            embedding_dim=params["embedding_dim"],
            dropout=params["dropout"],
            kan_hidden=params["kan_hidden"],
            grid_size=params["grid_size"],
            spline_order=params.get("spline_order", 3)
        )

    elif model_name == "RandomForest":
        return build_rf_model(params)

    elif model_name == "LightGBM":
        return build_lightgbm_model(params)

    elif model_name == "MLP":
        return build_mlp_model(input_size, num_classes, params)

    elif model_name == "CNN":
        return build_cnn_model(input_size, num_classes, params)

    else:
        raise ValueError(f"Model {model_name} not found.")