import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax.numpy as jnp
    import jax.scipy as scp
    import jax

    return jax, jnp, mo, scp


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
    chains = mh_sample(
        X=X,
        y=y,
        key=key,
        n_samples=100,
        burn_in=1000, 
        thin=5, 
        n_chains=1, 
        step_size=.1
    )

    chains
    return


if __name__ == "__main__":
    app.run()
