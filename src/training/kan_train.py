import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


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
            self.best_state = {k: v.cpu().clone() for k, v in
                               model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def evaluate_kan_model(model, data_loader, device):
    """
    Evaluates the model and collects its predictions and class probabilities.
    Returns accuracy and macro F1 alongside the prediction results.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.long().to(device)

            output = model(x)

            probs = F.softmax(output, dim=-1)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

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


def train_kan_optuna(model, train_loader, val_loader, device, config, y_train, lr):
    """
    Trains the KAN model during hyperparameter tuning.
    Uses class weights to account for imbalanced classes, along with
    a learning rate scheduler and early stopping.
    """
    y_flat = np.array(y_train).flatten()
    classes = np.unique(y_flat)
    weights = compute_class_weight('balanced', classes=classes, y=y_flat)
    weights = np.clip(weights, 0.1, 2.0)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    early_stopping = EarlyStopping(patience=config.get('patience', 5))

    model.to(device)
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.long().to(device)

            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        val_results = evaluate_kan_model(model, val_loader, device)
        val_macro_f1 = val_results['macro_f1']

        scheduler.step(val_macro_f1)
        if early_stopping(val_macro_f1, model):
            break

    # Load the weights from the epoch with the best validation score
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return model, early_stopping.best_f1


def train_kan_final(model, full_train_loader, device, config, y_full, lr):
    """
    Trains the final KAN model on the combined training and validation data.
    Uses class weights to help account for class imbalance and runs for
    the specified number of epochs without early stopping.
    """
    y_flat = np.array(y_full).flatten()
    classes = np.unique(y_flat)
    weights = compute_class_weight('balanced', classes=classes, y=y_flat)
    weights = np.clip(weights, 0.1, 2.0)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.to(device)
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        for x, y in full_train_loader:
            x = x.to(device)
            y = y.long().to(device)

            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    return model
