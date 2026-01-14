import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. SETTINGS ---
INPUT_SIZE = 2048  # ResNet50 feature size
HIDDEN_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 2. LOAD DATA ---
print("Loading features and labels...")
X = np.load("features.npy")
y = np.load("labels.npy")
NUM_CLASSES = len(np.unique(y)) 
print(f"Tespit edilen sınıf sayısı: {NUM_CLASSES}")
# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# --- 3. DEFINE MODEL ---
class FruitBrain(nn.Module):
    def __init__(self):
        super(FruitBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitBrain().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---

print(f"Starting training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
    
    accuracy = 100 * correct / len(y_test)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# --- 5. SAVE THE MODEL ---
torch.save(model.state_dict(), "fruit_classifier.pth")
print("\nModel saved as 'fruit_classifier.pth'")