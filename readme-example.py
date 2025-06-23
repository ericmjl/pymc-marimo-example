# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.14.6,<0.15",
#     "numba>=0.61.2,<0.62",
#     "nutpie @ git+https://github.com/pymc-devs/nutpie@main",
#     "pymc==4.2.0+1351.g02ffb7e8f",
# ]
# ///

import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # PyMC x Marimo

    This notebook is based on the linear regression example in the README of PyMC repo. Find the original example [here](https://github.com/pymc-devs/pymc?tab=readme-ov-file#linear-regression-example).
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    import pymc as pm

    return mo, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sampling from PyMC distributions

    Taking draws from a normal distribution

    Independent Variables:

    - Sunlight Hours: Number of hours the plant is exposed to sunlight daily.
    - Water Amount: Daily water amount given to the plant (in milliliters).
    - Soil Nitrogen Content: Percentage of nitrogen content in the soil.


    Dependent Variable:

    - Plant Growth (y): Measured as the increase in plant height (in centimeters) over a certain period.
    """
    )
    return


@app.cell
def _(pm):
    seed = 42
    x_dist = pm.Normal.dist(shape=(100, 3))
    x_data = pm.draw(x_dist, random_seed=seed)

    x_data[:5, :]
    return seed, x_data


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Defining Generative Model

    Define PyMC model without using `observed` keyword in any of the model distributions
    """
    )
    return


@app.cell
def _(pm, x_data):
    # Define coordinate values for all dimensions of the data
    coords = {
        "trial": range(100),
        "features": ["sunlight hours", "water amount", "soil nitrogen"],
    }

    # Define generative model
    with pm.Model(coords=coords) as generative_model:
        x = pm.Data("x", x_data, dims=["trial", "features"])

        # Model parameters
        betas = pm.Normal("betas", dims="features")
        sigma = pm.HalfNormal("sigma")

        # Linear model
        mu = x @ betas

        # Likelihood
        # Assuming we measure deviation of each plant from baseline
        plant_growth = pm.Normal("plant_growth", mu=mu, sigma=sigma, dims="trial")

    generative_model
    return coords, generative_model


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Fix parameters in the model

    Use the `pm.do` operator to fix any node in the model.
    """
    )
    return


@app.cell
def _(generative_model, pm):
    # Generating data from model by fixing parameters
    fixed_parameters = {
        "betas": [5, 20, 2],
        "sigma": 0.5,
    }
    synthetic_model = pm.do(generative_model, fixed_parameters)

    synthetic_model
    return (synthetic_model,)


@app.cell
def _(mo):
    mo.md(r"""Sample from the prior predictive with `pm.sample_prior_predictive`""")
    return


@app.cell
def _(pm, seed, synthetic_model):
    with synthetic_model:
        prior_idata = pm.sample_prior_predictive(random_seed=seed)
        synthetic_y = prior_idata.prior["plant_growth"].sel(draw=0, chain=0)

    synthetic_y
    return (synthetic_y,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Observed Random Variables

    Use the `pm.observe`
    """
    )
    return


@app.cell
def _(generative_model, pm, synthetic_y):
    # Infer parameters conditioned on observed data
    inference_model = pm.observe(generative_model, {"plant_growth": synthetic_y})

    inference_model
    return (inference_model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Bayesian Inference

    Use `pm.sample` .
    """
    )
    return


@app.cell
def _(inference_model, pm, seed):
    # Infer parameters conditioned on observed data
    with inference_model:
        idata = pm.sample(random_seed=seed, nuts_sampler="nutpie")
    return (idata,)


@app.cell
def _(idata, pm):
    summary = pm.stats.summary(idata, var_names=["betas", "sigma"])

    summary
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Out of sample predictions

    The inference model can
    """
    )
    return


@app.cell
def _(coords, inference_model, pm, seed):
    # Simulate new data conditioned on inferred parameters
    new_x_data = pm.draw(
        pm.Normal.dist(shape=(3, 3)),
        random_seed=seed,
    )
    new_coords = coords | {"trial": [0, 1, 2]}

    with inference_model:
        pm.set_data({"x": new_x_data}, coords=new_coords)

    inference_model
    return


@app.cell
def _(idata, inference_model, pm, seed):
    with inference_model:
        pm.sample_posterior_predictive(
            idata,
            predictions=True,
            extend_inferencedata=True,
            random_seed=seed,
        )

    pm.stats.summary(idata.predictions, kind="stats")
    return


@app.cell
def _(mo):
    mo.md(r"""## Counterfactual""")
    return


@app.cell
def _(inference_model, pm):
    # Simulate new data, under a scenario where the first beta is zero
    plant_growth_model = pm.do(
        inference_model,
        {inference_model["betas"]: inference_model["betas"] * [0, 1, 1]},
    )

    plant_growth_model
    return (plant_growth_model,)


@app.cell
def _(idata, plant_growth_model, pm, seed):
    with plant_growth_model:
        new_predictions = pm.sample_posterior_predictive(
            idata,
            predictions=True,
            random_seed=seed,
        )

    pm.stats.summary(new_predictions, kind="stats")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
