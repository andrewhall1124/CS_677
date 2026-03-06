import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax.numpy as jnp
    import jax.scipy as scp
    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    import pymc as pm
    import arviz as az

    return jax, jnp, mo, np, plt, pm, scp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Simulate Data
    """)
    return


@app.cell
def _(jax):
    key = jax.random.PRNGKey(42)
    return (key,)


@app.cell
def _(jax, jnp, key):
    def generate_data(n_samples: int, key: jax.Array) -> tuple[jnp.array, jnp.array]:
        w = jnp.array([0.5, -1.2, 3.0])
        b = 1.8
        sigma_y = 1.0

        key, subkey1, subkey2 = jax.random.split(key, 3)
        X = jax.random.uniform(key=subkey1, minval=-3, maxval=3, shape=(n_samples, 3))
        mu = X @ w + b
        y = jax.random.normal(key=subkey2, shape=(n_samples,)) * sigma_y + mu

        return X, y

    X, y = generate_data(1000, key)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Specify Model
    """)
    return


@app.cell
def _(jnp, scp):
    def log_likelihood(w: jnp.array, b: float, X: jnp.array, y: jnp.array) -> float:
        mu = X @ w + b
        return jnp.sum(scp.stats.norm.logpdf(y, loc=mu, scale=1.0))

    def log_prior(w: jnp.array, b: float, sigma_w: float = 10.0, sigma_b: float = 10.0) -> float:
        lp_w = jnp.sum(scp.stats.norm.logpdf(w, loc=0, scale=sigma_w))
        lp_b = scp.stats.norm.logpdf(b, loc=0, scale=sigma_b)
        return lp_w + lp_b

    def log_posterior(w: jnp.array, b: float, X: jnp.array, y: jnp.array) -> float:
        return log_likelihood(w, b, X, y) + log_prior(w, b)

    return (log_posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Implement MCMC
    """)
    return


@app.cell
def _(jax, jnp, log_posterior):
    def mh_sample(
        X: jnp.array,
        y: jnp.array,
        key: jax.Array,
        n_samples: int = 10000,
        burn_in: int = 1000,
        thin: int = 5,
        n_chains: int = 4,
        step_size: float = 0.1,
    ) -> list[jnp.array]:
        chains = []

        for chain in range(n_chains):
            # Split key for this chain
            key, chain_key = jax.random.split(key)

            # Random starting location per chain
            chain_key, subkey1, subkey2 = jax.random.split(chain_key, 3)
            w = jax.random.normal(key=subkey1, shape=(3,))
            b = jax.random.normal(key=subkey2)
            log_p = log_posterior(w, b, X, y)

            samples = []

            for i in range(n_samples * thin + burn_in):
                # Propose: random walk with Gaussian perturbation
                chain_key, subkey1, subkey2, subkey3 = jax.random.split(chain_key, 4)
                w_prop = w + jax.random.normal(key=subkey1, shape=(3,)) * step_size
                b_prop = b + jax.random.normal(key=subkey2) * step_size

                log_p_prop = log_posterior(w_prop, b_prop, X, y)

                # Accept/reject in log space: log(u) < log_p_prop - log_p
                log_u = jnp.log(jax.random.uniform(key=subkey3))
                if log_u < log_p_prop - log_p:
                    w, b = w_prop, b_prop
                    log_p = log_p_prop

                # Collect after burn-in, respecting thinning
                if i >= burn_in and (i - burn_in) % thin == 0:
                    samples.append(jnp.concatenate([w, jnp.array([b])]))

            chains.append(jnp.stack(samples))

        return chains

    return (mh_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MCMC Diagnostics
    """)
    return


@app.cell
def _(X, key, mh_sample, y):
    step_size = 0.05
    chains = mh_sample(
        X=X,
        y=y,
        key=key,
        n_samples=5000,
        burn_in=2000,
        thin=5,
        n_chains=4,
        step_size=step_size,
    )
    return chains, step_size


@app.cell
def _(chains, np):
    param_names = [r"$w_0$", r"$w_1$", r"$w_2$", r"$b$"]
    true_params = [0.5, -1.2, 3.0, 1.8]
    all_samples = np.concatenate([np.array(c) for c in chains], axis=0)
    return all_samples, param_names, true_params


@app.cell
def _(chains, np, param_names, plt, step_size, true_params):
    def trace_plots():
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        # PLot a chart for each parameter
        for i, (ax, name, true_val) in enumerate(zip(axes, param_names, true_params)):

            # Plot each chain for the parameter
            for j, chain in enumerate(chains):
                ax.plot(np.array(chain[:, i]), label=f"Chain {j+1}")

            # True parameter
            ax.axhline(true_val, color="red", linestyle='--', label="True")
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)

        # Format
        axes[-1].set_xlabel("Sample")
        fig.suptitle(f"Trace Plots (step size = {step_size})")
        plt.tight_layout()
        plt.show()

    trace_plots()
    return


@app.cell
def _(all_samples, np, param_names, plt, true_params):
    def posterior_histograms():    
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        # Plot each parameter's posterior histogram
        for i, (ax, name, true_val) in enumerate(zip(axes.flat, param_names, true_params)):
            # Get samples
            samples_i = all_samples[:, i]

            # Get credible intervals
            lo, hi = np.percentile(samples_i, [2.5, 97.5])

            # Plot each histogram
            ax.hist(samples_i, bins=50)
            ax.axvline(true_val, color="red", linestyle="--", label=f"True = {true_val}")
            ax.axvline(lo, color="orange", linestyle="--", label=f"2.5% = {lo:.3f}")
            ax.axvline(hi, color="orange", linestyle="--", label=f"97.5% = {hi:.3f}")
            ax.set_title(f"Posterior of {name}")
            ax.legend(fontsize=7)

        # Format
        plt.tight_layout()
        plt.show()

    posterior_histograms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. PyMC Implementation
    """)
    return


@app.cell
def _(jnp, np, pm):
    def pymc_mh_sample(
        X: jnp.array,
        y: jnp.array,
        n_samples: int = 10000,
        burn_in: int = 1000,
        thin: int = 5,
        n_chains: int = 4,
        step_size: float = 0.1,
    ) -> list[jnp.array]:   
        X_np = np.array(X)
        y_np = np.array(y)
    
        with pm.Model() as linear_model:
            # Priors
            w = pm.Normal("w", mu=0, sigma=10, shape=3)
            b = pm.Normal("b", mu=0, sigma=10)
            sigma_y = pm.HalfNormal("sigma_y", sigma=5)
    
            # Likelihood
            mu = X_np @ w + b
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_np)
    
            # Sample (NUTS by default)
            trace = pm.sample(n_samples, tune=thin, chains=n_chains)

        return trace

    return (pymc_mh_sample,)


@app.cell
def _(X, pymc_mh_sample, step_size, y):
    trace = pymc_mh_sample(
        X=X,
        y=y,
        n_samples=5000,
        burn_in=2000,
        thin=5,
        n_chains=4,
        step_size=step_size,
    )
    return (trace,)


@app.cell
def _(param_names, true_params):
    sigma_y_true = 1.0
    all_true = true_params + [sigma_y_true]
    all_names = param_names + [r"$\sigma_y$"]
    return all_names, all_true


@app.cell
def _(all_names, all_true, np, plt, trace):
    def pymc_posterior_histograms():
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        axes = axes.flatten()
    
        param_map = [("w", 0), ("w", 1), ("w", 2), ("b", None), ("sigma_y", None)]
    
        for i, (name, idx) in enumerate(param_map):
            ax = axes[i]
            if idx is not None:
                samples = trace.posterior[name].values[:, :, idx].flatten()
            else:
                samples = trace.posterior[name].values.flatten()
    
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    
            ax.hist(samples, bins=50)
            ax.axvline(all_true[i], color="red", linestyle="--", label=f"True = {all_true[i]}")
            ax.axvline(ci_low, color="orange", linestyle="--", label=f"95% CI [{ci_low:.2f}, {ci_high:.2f}]")
            ax.axvline(ci_high, color="orange", linestyle="--")
            ax.set_title(all_names[i])
            ax.legend(fontsize=8)
    
        axes[-1].set_visible(False)
        plt.suptitle("PyMC Posteriors with 95% Credible Intervals")
        plt.tight_layout()
        plt.show()

    pymc_posterior_histograms()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
