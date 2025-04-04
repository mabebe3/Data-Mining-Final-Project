import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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

        # Output layer
        self.output_layer = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Hyperparameters (set your desired values here)
num_layers = 3
hidden_units = 512
dropout_rate = 0.3
lr = 1e-4
weight_decay = 0
batch_size = 1024
patience = 5
max_epochs = 50

# Assuming you have your data in numpy arrays
# X_train, y_train, X_test, y_test = ...

# Convert data to PyTorch tensors and create datasets
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create training and validation datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

# Initialize model, optimizer, and loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HighDimNet(X_train.shape[1], num_layers=num_layers, 
                  hidden_units=hidden_units, dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Training loop with early stopping
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val.view(-1, 1))
            val_loss += loss.item() * X_val.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{max_epochs}')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

# Load best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Test evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        loss = criterion(outputs, y_test.view(-1, 1))
        test_loss += loss.item() * X_test.size(0)
test_loss /= len(test_loader.dataset)

print(f'Final Test Loss: {test_loss:.4f}')