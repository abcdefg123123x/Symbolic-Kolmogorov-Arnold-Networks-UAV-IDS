from sklearn.metrics import f1_score, accuracy_score


def _preprocess_rf_data(X, y=None):
    """Converts the input data to NumPy arrays and reshapes it if needed."""
    if hasattr(X, "cpu"):
        X = X.cpu().numpy()
    if y is not None and hasattr(y, "cpu"):
        y = y.cpu().numpy()

    # Reshape sequence data into a 2D array for sklearn
    if len(X.shape) == 3:
        n_samples, seq_len, feat_dim = X.shape
        X = X.reshape(n_samples, seq_len * feat_dim)

    if y is not None and len(y.shape) > 1:
        y = y.reshape(-1)

    return X, y


def evaluate_rf_model(model, X_val, y_val):
    """
    Evaluates the Random Forest model and returns its predictions,
    class probabilities and performance metrics.
    """
    X_val_np, y_val_np = _preprocess_rf_data(X_val, y_val)

    preds = model.predict(X_val_np)
    probs = model.predict_proba(X_val_np) if hasattr(model,
                                                     "predict_proba") else None

    acc = accuracy_score(y_val_np, preds)
    macro_f1 = f1_score(y_val_np, preds, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'predictions': preds,
        'probabilities': probs,
        'true_labels': y_val_np
    }


def train_rf_optuna(model, X_train, y_train, X_val, y_val):
    """
    Fits the Random Forest model on the training data and checks
    its performance on the validation set during hyperparameter tuning.
    """
    X_train_np, y_train_np = _preprocess_rf_data(X_train, y_train)

    model.fit(X_train_np, y_train_np)

    val_results = evaluate_rf_model(model, X_val, y_val)
    val_macro_f1 = val_results['macro_f1']

    return model, val_macro_f1


def train_rf_final(model, X_full, y_full):
    """
    Fits the final Random Forest model using the combined training
    and validation data.
    """
    X_full_np, y_full_np = _preprocess_rf_data(X_full, y_full)

    model.fit(X_full_np, y_full_np)

    return model
