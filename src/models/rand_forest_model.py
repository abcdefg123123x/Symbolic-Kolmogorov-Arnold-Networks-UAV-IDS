from sklearn.ensemble import RandomForestClassifier


def build_rf_model(params):
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **params
    )