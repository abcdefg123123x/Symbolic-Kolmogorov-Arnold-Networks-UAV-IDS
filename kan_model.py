import torch.nn as nn
from efficient_kan import KAN as EfficientKAN


class KANModel(nn.Module):
    """
    Pure Kolmogorov-Arnold Network (KAN) for tabular data classification.
    """

    def __init__(self, input_size, num_classes, embedding_dim,
                 dropout, kan_hidden, grid_size, spline_order):
        super().__init__()

        # Input projection matching tabular feature space
        self.input_projection = nn.Linear(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Pure KAN backbone mapping embedding_dim -> kan_hidden -> num_classes
        self.kan = EfficientKAN(
            [embedding_dim, kan_hidden, num_classes],
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x):
        """
        Forward pass for 2D tabular data.
        Args:
            x: Tensor of shape (batch_size, input_size)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Project and normalize input features
        x = self.input_projection(x)
        x = self.dropout(self.layer_norm(x))

        # Pass directly through the KAN layers
        logits = self.kan(x)

        return logits