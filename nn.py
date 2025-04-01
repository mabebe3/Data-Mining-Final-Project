import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping

# Define the neural network architecture
class HighDimNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, hidden_units=512, 
                 dropout_rate=0.3, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_units))
        self.layers.append(activation())
        self.layers.append(nn.BatchNorm1d(hidden_units))
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(activation())
            self.layers.append(nn.BatchNorm1d(hidden_units))
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer (modify for classification/regression)
        self.output_layer = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Wrap model in skorch for sklearn compatibility
model = NeuralNetRegressor(
    module=HighDimNet,
    module__input_dim=X_train.shape[1],  # Set input dimension
    criterion=nn.MSELoss,
    max_epochs=50,
    batch_size=1024,
    optimizer=optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

# Define hyperparameter grid
param_grid = {
    'module__num_layers': [3, 4, 5],
    'module__hidden_units': [256, 512, 1024],
    'module__dropout_rate': [0.2, 0.3, 0.4],
    'optimizer__lr': [1e-4, 3e-4, 1e-3],
    'optimizer__weight_decay': [0, 1e-5],
    'batch_size': [512, 1024],
}

# Initialize grid search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=1,  # Reduce if memory-constrained
    verbose=2
)

# Assuming you have your data in numpy arrays
# X_train, y_train, X_test, y_test = ...

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

# Run grid search
grid.fit(X_train_tensor, y_train_tensor)

# Best results
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Evaluate on test set
best_model = grid.best_estimator_
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
test_score = best_model.score(X_test_tensor, y_test_tensor)
print(f"Test score: {test_score:.4f}")