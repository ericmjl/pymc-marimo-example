# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==0.55.0",
#     "arviz==0.21.0",
#     "marimo>=0.14.6,<0.15",
#     "matplotlib==3.10.3",
#     "numba>=0.61.2,<0.62",
#     "nutpie @ git+https://github.com/pymc-devs/nutpie@main",
#     "pymc==4.2.0+1351.g02ffb7e8f",
# ]
# ///

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # PyMC x Marimo

    This notebook is based on the linear regression example in the README of PyMC repo. Find the original example [here](https://github.com/pymc-devs/pymc?tab=readme-ov-file#linear-regression-example).

    There are independent variables:

    - Sunlight Hours: Number of hours the plant is exposed to sunlight daily.
    - Water Amount: Daily water amount given to the plant (in milliliters).
    - Soil Nitrogen Content: Percentage of nitrogen content in the soil.

    as well as the dependent variable:

    - Plant Growth (y): Measured as the increase in plant height (in centimeters) over a certain period.

    The functional form of the model is:

    $$
    y_i \sim \alpha + \mathbf{X}\beta + \epsilon_i
    $$

    with

    $$
    \epsilon_i \sim \mathcal{N}(0, \sigma^2)
    $$
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    import arviz as az
    import pymc as pm

    import matplotlib.pyplot as plt

    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100

    return mo, pm, az, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sampling from PyMC distributions

    We will generate $\mathbf{X}$ by taking draws from a normal distribution.

    PyMC distribution outside of modelcontext can be created with the `dist` classmethod. This returns a `TensorVariable` which is from the `pytensor` package.

    The `pm.draw` function can be used to generate samples from any `TensorVariable`.
    """
    )
    return


@app.cell
def _(pm):
    seed = sum(map(ord, "PyMC x Marimo"))
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
        pm.Normal("plant_growth", mu=mu, sigma=sigma, dims="trial")

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
def _(mo):
    def beta_slider(value: float):
        return mo.ui.slider(start=-30, value=value, stop=30, show_value=True, step=0.01)

    fixed_betas = mo.ui.dictionary(
        {
            "sunlight hours": beta_slider(5),
            "water amount": beta_slider(20),
            "soil nitrogen": beta_slider(2),
        }
    )

    fixed_sigma = mo.ui.slider(
        start=0.01,
        value=0.5,
        stop=5,
        show_value=True,
        step=0.01,
    )

    mo.vstack(
        [
            fixed_betas,
            fixed_sigma,
        ]
    )
    return fixed_betas, fixed_sigma


@app.cell
def _(mo):
    run_following_cells = mo.ui.button(label="Run Following Cells")

    run_following_cells
    return (run_following_cells,)


@app.cell
def _(run_following_cells):
    # This isn't working the way I want at the moment
    run_following_cells.value
    return


@app.cell
def _(fixed_betas, fixed_sigma, generative_model, mo, pm, run_following_cells):
    mo.stop(not run_following_cells.value)

    # Generating data from model by fixing parameters
    fixed_parameters = {
        "betas": [*fixed_betas.value.values()],
        "sigma": fixed_sigma.value,
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
        prior_idata = pm.sample_prior_predictive(
            random_seed=seed, compile_kwargs={"mode": "NUMBA"}
        )
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

    The `inference_model` includes the `x` with shape `(100, 3)`. However, we can use `pm.set_data` to change the `x` to a new value.

    After updating `x`, we can see both `x` and observed `plant_growth` have changed in the graph.
    """
    )
    return


@app.cell
def _(coords, inference_model, pm, seed):
    # Simulate new data conditioned on inferred parameters
    new_x_data = pm.draw(
        pm.Normal.dist(shape=(4, 3)),
        random_seed=seed,
    )
    new_coords = coords | {"trial": [0, 1, 2, 3]}

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
    mo.md(
        r"""
    ## Counterfactual

    The `pm.do` operator can be used again to simulate new data under a scenario where the first beta is zero
    """
    )
    return


@app.cell
def _(inference_model, pm):
    mask = [0, 1, 1]
    plant_growth_model = pm.do(
        inference_model,
        {inference_model["betas"]: inference_model["betas"] * mask},
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
def _(mo):
    mo.md(r"""## InferenceData Posterior Explorer""")
    return


@app.cell(hide_code=True)
def _(idata, mo):
    vars = list(idata.posterior.data_vars)
    variable_select = mo.ui.dropdown(
        label="Select random var to view",
        options=vars,
        value=vars[0],
    )

    plot_type = ["posterior", "forest", "trace"]
    plot_type_select = mo.ui.dropdown(
        label="Select plot type",
        options=plot_type,
        value=plot_type[0],
    )

    mo.vstack(
        [
            variable_select,
            plot_type_select,
        ]
    )
    return plot_type_select, variable_select


@app.cell(hide_code=True)
def _(az, plt, idata, plot_type_select, variable_select):
    if plot_type_select.value == "posterior":
        _ = az.plot_posterior(idata, var_names=[variable_select.value])
    elif plot_type_select.value == "trace":
        _ = az.plot_trace(idata, var_names=[variable_select.value])
    else:
        _ = az.plot_forest(idata, var_names=[variable_select.value])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## InferenceData Diagnostics Explorer""")
    return


@app.cell
def _(idata):
    idata
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
