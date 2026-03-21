import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Build model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class VanillaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
    
model = VanillaNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)


# Training
train = pd.read_csv("data/train-1.csv")
X_train = train['X']
y_train = train['y']

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

for epoch in range(1000):
    y_hat = model(X_train)

    loss = criterion(y_hat, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Testing
test = pd.read_csv("data/test-1.csv")
X_test = test['X']
y_test = test['y']

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    y_pred = model(X_test)

pred = pd.DataFrame({
    'X': X_test.cpu().squeeze().numpy(),
    'y': y_pred.cpu().squeeze().numpy()
})

lb, ub = train['X'].min(), train['X'].max()

sns.scatterplot(train, x='X', y='y', label='TRAIN', alpha=.5)
sns.scatterplot(test, x='X', y='y', label='TEST', alpha=.5)
sns.lineplot(pred, x='X', y='y', label='Predicitions', color='green')

plt.axvline(x=lb, color='red', linestyle='--', label='Training Data Ends')
plt.axvline(x=ub, color='red', linestyle='--', label='Training Data Ends')

plt.legend()

plt.savefig("results/vanilla_nn.png", dpi=300)