import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score


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


def evaluate_cnn_model(model, data_loader, device):
    """
    Evaluates the CNN on the given dataset and calculates accuracy and macro F1.
    Also collects the predictions and class probabilities for later analysis.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            probs = F.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(y_batch.view(-1).cpu().numpy())
            all_probs.append(probs.view(-1, probs.size(-1)).cpu().numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(labels_arr, preds_arr)
    macro_f1 = f1_score(labels_arr, preds_arr, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'predictions': preds_arr,
        'probabilities': probs_arr,
        'true_labels': labels_arr
    }


def train_cnn_optuna(model, train_loader, val_loader, device, config, lr):
    """
    Trains the CNN during hyperparameter tuning.
    Uses gradient clipping and early stopping based on validation macro F1.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=config.get('patience', 5))
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_results = evaluate_cnn_model(model, val_loader, device)
        val_macro_f1 = val_results['macro_f1']

        if early_stopping(val_macro_f1, model):
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return model, early_stopping.best_f1


def train_cnn_final(model, full_train_loader, device, config, lr):
    """
    Trains the final CNN using the combined training and validation data.
    Runs for the specified number of epochs without checking validation performance.
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model
