import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

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

for epoch in trange(1000):
    y_hat = model(X_train)

    loss = criterion(y_hat, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Testing
test = pd.read_csv("data/test-1.csv")
X_test = torch.tensor(test['X'], dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(test['y'], dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    y_test_pred = model(X_test)
    rmse = torch.sqrt(torch.mean((y_test_pred - y_test) ** 2)).item()

# Predictions over linspace
lb, ub = test['X'].min(), test['X'].max()
X_lin = torch.linspace(lb, ub, 200).unsqueeze(1).to(device)

with torch.no_grad():
    y_lin = model(X_lin)

pred = pd.DataFrame({
    'X': X_lin.cpu().squeeze().numpy(),
    'y': y_lin.cpu().squeeze().numpy()
})

train_lb, train_ub = train['X'].min(), train['X'].max()

sns.scatterplot(train, x='X', y='y', label='TRAIN', alpha=.5)
sns.scatterplot(test, x='X', y='y', label='TEST', alpha=.5)
sns.lineplot(pred, x='X', y='y', label='Predictions', color='green')

plt.axvline(x=train_lb, color='red', linestyle='--')
plt.axvline(x=train_ub, color='red', linestyle='--', label='Training Data Ends')

plt.suptitle("Vanilla NN")
plt.title(f"RMSE (test): {rmse:.4f}")
plt.legend()

plt.savefig("results/vanilla_nn.png", dpi=300)