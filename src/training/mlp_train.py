import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = 0.0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_f1, model):
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def evaluate_mlp_model(model, data_loader, device):
    """
    Evaluates the model and collects its predictions and class probabilities.
    Returns the accuracy and macro F1 score along with the results.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    acc = accuracy_score(labels_arr, preds_arr)
    macro_f1 = f1_score(labels_arr, preds_arr, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'predictions': preds_arr,
        'probabilities': probs_arr,
        'true_labels': labels_arr
    }


def train_mlp_optuna(model, train_loader, val_loader, device, config, lr):
    """
    Trains the MLP during hyperparameter tuning.
    Checks validation performance after each epoch and stops training
    if the macro F1 score stops improving.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=config.get('patience', 5))

    for epoch in range(config["training"]["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        val_results = evaluate_mlp_model(model, val_loader, device)
        val_macro_f1 = val_results['macro_f1']

        if early_stopping(val_macro_f1, model):
            break

    # Load the weights from the epoch with the best validation score
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return model, early_stopping.best_f1


def train_mlp_final(model, full_train_loader, device, config, lr):
    """
    Trains the final MLP using the combined training and validation data.
    Runs for the configured number of epochs without checking validation
    performance or using early stopping.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in full_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model
