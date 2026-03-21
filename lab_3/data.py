import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data/train-1.csv")
test = pd.read_csv("data/test-1.csv")

lb, ub = train['X'].min(), train['X'].max()

sns.scatterplot(train, x='X', y='y', label='TRAIN', alpha=.5)
sns.scatterplot(test, x='X', y='y', label='TEST', alpha=.5)

plt.axvline(x=lb, color='red', linestyle='--', label='Training Data Ends')
plt.axvline(x=ub, color='red', linestyle='--', label='Training Data Ends')

plt.legend()

plt.savefig("results/data.png", dpi=300)