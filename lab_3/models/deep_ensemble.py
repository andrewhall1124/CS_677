import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

# Build model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class EnsembleMember(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out = self.model(x)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        return mu, log_var


def nll_loss(mu, log_var, y):
    return torch.mean(0.5 * torch.exp(-log_var) * (y - mu) ** 2 + 0.5 * log_var)


# Training data
train = pd.read_csv("data/train-1.csv")
X_train = torch.tensor(train['X'], dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(train['y'], dtype=torch.float32).unsqueeze(1).to(device)

# Train M ensemble members
M = 5
models = []

for m in range(M):
    model = EnsembleMember().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in trange(1000, desc=f"Training model {m+1}/{M}"):
        mu, log_var = model(X_train)
        loss = nll_loss(mu, log_var, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    models.append(model)

# Testing
test = pd.read_csv("data/test-1.csv")
X_test = torch.tensor(test['X'], dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(test['y'], dtype=torch.float32).unsqueeze(1).to(device)

# Collect predictions from all ensemble members
with torch.no_grad():
    all_mu = []
    all_var = []
    for model in models:
        mu, log_var = model(X_test)
        all_mu.append(mu)
        all_var.append(torch.exp(log_var))

    all_mu = torch.stack(all_mu)     # (M, N, 1)
    all_var = torch.stack(all_var)   # (M, N, 1)

    mu_star = all_mu.mean(dim=0)
    var_star = (all_var + all_mu ** 2).mean(dim=0) - mu_star ** 2

    rmse = torch.sqrt(torch.mean((mu_star - y_test) ** 2)).item()

# Predictions over linspace
X_lin = torch.linspace(test['X'].min(), test['X'].max(), 200).unsqueeze(1).to(device)

with torch.no_grad():
    all_mu = []
    all_var = []
    for model in models:
        mu, log_var = model(X_lin)
        all_mu.append(mu)
        all_var.append(torch.exp(log_var))

    all_mu = torch.stack(all_mu)
    all_var = torch.stack(all_var)

    mu_star = all_mu.mean(dim=0)
    var_star = (all_var + all_mu ** 2).mean(dim=0) - mu_star ** 2

upper = (mu_star + 1.96 * torch.sqrt(var_star)).cpu().squeeze().numpy()
lower = (mu_star - 1.96 * torch.sqrt(var_star)).cpu().squeeze().numpy()

pred = pd.DataFrame({
    'X': X_lin.cpu().squeeze().numpy(),
    'y': mu_star.cpu().squeeze().numpy()
})

# Plotting
lb, ub = train['X'].min(), train['X'].max()

sns.scatterplot(train, x='X', y='y', label='TRAIN', alpha=.5)
sns.scatterplot(test, x='X', y='y', label='TEST', alpha=.5)
sns.lineplot(pred, x='X', y='y', label='Predictions', color='green')
plt.fill_between(
    pred['X'].values, lower, upper,
    alpha=0.3, color='green', label='95% CI'
)

plt.axvline(x=lb, color='red', linestyle='--')
plt.axvline(x=ub, color='red', linestyle='--', label='Training Data Ends')

plt.suptitle("Deep Ensemble")
plt.title(f"RMSE (test): {rmse:.4f}")
plt.legend()

plt.savefig("results/deep_ensemble.png", dpi=300)
