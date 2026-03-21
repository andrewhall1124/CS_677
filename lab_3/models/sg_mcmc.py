import torch
from torch import nn
from torch.func import functional_call
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import posteriors
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

def log_posterior(params, batch, prior_sd=5.0, noise_var=0.25):
    x, y, n_data = batch
    y_hat = functional_call(model, params, (x,))

    log_lik = -0.5 * ((y - y_hat) ** 2) / noise_var
    log_lik = log_lik.sum() * (n_data / x.shape[0])

    log_prior = sum(
        (-0.5 * (p ** 2) / prior_sd ** 2).sum() for p in params.values()
    )

    return log_lik + log_prior, {}

transform = posteriors.sgmcmc.sgld.build(
    log_posterior=log_posterior,
    lr=5e-5,
    temperature=0.01,
)

state = transform.init(params=dict(model.named_parameters()))

# Training
train = pd.read_csv("data/train-1.csv")
X_train = torch.tensor(train['X'], dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(train['y'], dtype=torch.float32).unsqueeze(1).to(device)
n_data = X_train.shape[0]

for epoch in trange(10000):
    state, _ = transform.update(
        state,
        batch=(X_train, y_train, n_data),
    )

# Collect posterior samples
n_samples = 100

# RMSE on test set
test = pd.read_csv("data/test-1.csv")
X_test = torch.tensor(test['X'], dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(test['y'], dtype=torch.float32).unsqueeze(1).to(device)

test_preds = []
for _ in range(n_samples):
    state, _ = transform.update(state, batch=(X_train, y_train, n_data))
    with torch.no_grad():
        y_hat = functional_call(model, state.params, (X_test,))
    test_preds.append(y_hat)

test_preds = torch.stack(test_preds)
test_mu = test_preds.mean(dim=0)
rmse = torch.sqrt(torch.mean((test_mu - y_test) ** 2)).item()

# Predictions over linspace
X_lin = torch.linspace(test['X'].min(), test['X'].max(), 200).unsqueeze(1).to(device)

preds = []
for _ in range(n_samples):
    state, _ = transform.update(state, batch=(X_train, y_train, n_data))
    with torch.no_grad():
        y_hat = functional_call(model, state.params, (X_lin,))
    preds.append(y_hat)

preds = torch.stack(preds)
mu = preds.mean(dim=0)
var = preds.var(dim=0) + 0.25

upper = (mu + 1.96 * torch.sqrt(var)).cpu().squeeze().numpy()
lower = (mu - 1.96 * torch.sqrt(var)).cpu().squeeze().numpy()

pred = pd.DataFrame({
    'X': X_lin.cpu().squeeze().numpy(),
    'y': mu.cpu().squeeze().numpy()
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

plt.suptitle("SG-MCMC (SGLD)")
plt.title(f"RMSE (test): {rmse:.4f}")
plt.legend()

plt.savefig("results/sg_mcmc.png", dpi=300)
