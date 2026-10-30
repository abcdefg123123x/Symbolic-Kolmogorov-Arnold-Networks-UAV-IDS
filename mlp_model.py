import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dims, dropout):
        super(MLP, self).__init__()

        layers = []
        in_dim = input_size

        # Dynamically create hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_mlp_model(input_size, num_classes, params):
    # Extract how many layers wanted
    num_layers = params["num_layers"]

    # Slice the list to only use the number of layers Optuna requested
    # If num_layers is 1, only take the first element
    hidden_dims = [params[f"layer_{i}_size"] for i in range(1, num_layers + 1)]

    return MLP(
        input_size=input_size,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=params["dropout"]
    )