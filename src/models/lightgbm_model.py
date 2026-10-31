import lightgbm as lgb


def build_lightgbm_model(params):
    """Initialises and returns a LightGBM Classifier matching project conventions."""
    return lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
        **params
    )