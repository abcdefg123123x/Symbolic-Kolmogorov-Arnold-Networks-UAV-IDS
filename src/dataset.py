
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# ============================================================
# Custom PyTorch Dataset
# ============================================================

class TorchDataset(Dataset):
    """Converts tabular data into tensors that can be used by PyTorch."""

    def __init__(self, features, targets):
        self.features = torch.tensor(features.copy(), dtype=torch.float32)
        self.targets = torch.tensor(targets.copy(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ============================================================
# Data preprocessing helpers
# ============================================================

def _encode_labels(df, target_column, binary_mode, benign_label):
    """
    Converts the target labels into numerical values.

    In binary classification, all labels are grouped into either
    Benign or Malicious. Otherwise, each original class is encoded
    separately for multi-class classification.
    """
    encoder = LabelEncoder()

    if binary_mode:
        df["temp_label"] = df[target_column].apply(
            lambda label: "Benign" if label == benign_label else "Malicious"
        )

        df["Label_encoded"] = encoder.fit_transform(df["temp_label"])
        df = df.drop(columns=["temp_label"])

    else:
        df["Label_encoded"] = encoder.fit_transform(df[target_column])

    return df, encoder


def _encode_categorical_features(df, config_cat_cols):
    """
    Converts the categorical columns specified in the config
    into numerical features using one-hot encoding.
    """
    cat_cols = [col for col in config_cat_cols if col in df.columns]

    if cat_cols:
        print(f"Encoding categorical columns: {cat_cols}")

        # Convert values to strings first to avoid mixed data types.
        df[cat_cols] = df[cat_cols].astype(str)

        df = pd.get_dummies(
            df,
            columns=cat_cols,
            drop_first=True,
            dtype=int
        )

    return df


def _scale_features(X_train, X_val, X_test):
    """
    Standardises the feature values using statistics from the
    training set only, then applies the same transformation
    to the validation and test sets.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


def _remove_constant_columns(df):
    """Removes columns that contain only a single unique value."""

    constant_cols = [
        col for col in df.columns
        if df[col].nunique() <= 1
    ]

    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df


# ============================================================
# Train, validation and test splitting
# ============================================================

def _random_stratified_split(X, y):
    """
    Splits the dataset into training, validation and test sets.

    Stratification keeps the class distribution relatively
    consistent across all three sets.
    """
    # First, separate 70% of the data for training.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        shuffle=True,
        random_state=100
    )

    # Divide the remaining 30% into validation (10%) and test (20%).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=2 / 3,
        stratify=y_temp,
        shuffle=True,
        random_state=100
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Main dataset loading function
# ============================================================

def load_tabular_dataset(config):
    """
    Loads a tabular dataset and prepares it for model training.

    The preprocessing steps include:
    - Removing missing and constant values.
    - Encoding the target labels.
    - Handling categorical features.
    - Applying optional class-wise sampling.
    - Splitting the data into train, validation and test sets.
    - Standardising the feature values.

    Returns the processed datasets along with the fitted
    label encoder and scaler.
    """
    data_cfg = config["data"]

    # Load the dataset and remove rows containing missing values.
    df = pd.read_csv(data_cfg["path"]).dropna()

    # Remove columns that do not provide any useful variation.
    df = _remove_constant_columns(df)

    # Timestamps are not used directly as model features.
    timestamp_col = data_cfg.get("timestamp_column")

    if timestamp_col and timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])

    # Convert the target labels into numerical class IDs.
    df, encoder = _encode_labels(
        df,
        data_cfg["target_column"],
        data_cfg["binary_mode"],
        data_cfg["benign_label"]
    )

    # Keep target and metadata columns out of the feature set.
    excluded = [
        data_cfg["target_column"],
        "Label_encoded"
    ] + data_cfg.get("meta_columns", [])

    if timestamp_col and timestamp_col in df.columns:
        excluded.append(timestamp_col)

    # Encode categorical features specified in the configuration.
    config_cat_cols = data_cfg.get("categorical_features", [])

    df = _encode_categorical_features(
        df,
        config_cat_cols=config_cat_cols
    )

    # Optionally reduce the dataset size while keeping samples
    # from each class.
    sample_pct = data_cfg["sample_pct"]

    if sample_pct < 1.0:
        sampled_data = []

        for label in df["Label_encoded"].unique():
            class_data = df[df["Label_encoded"] == label]

            # Select the requested proportion of each class.
            sample_size = int(len(class_data) * sample_pct)
            sampled_data.append(class_data.iloc[:sample_size])

        df = pd.concat(sampled_data).reset_index(drop=True)

    # Separate the input features from the target labels.
    X = df[[col for col in df.columns if col not in excluded]]
    y = df["Label_encoded"]

    # Create stratified training, validation and test sets.
    X_train, X_val, X_test, y_train, y_val, y_test = (
        _random_stratified_split(X, y)
    )

    # Fit the scaler on training data and reuse it for the
    # validation and test sets to avoid data leakage.
    X_train, X_val, X_test, scaler = _scale_features(
        X_train,
        X_val,
        X_test
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train.values,
        y_val.values,
        y_test.values,
        encoder,
        scaler
    )
