# PyMC Marimo Example

> [!CAUTION]
>
> This example is still a work in progress.

This is duplicate of the PyMC README example but using marimo to run the notebook. Check out the original example [here](https://github.com/pymc-devs/pymc?tab=readme-ov-file#linear-regression-example).

This example uses [`nutpie`](https://github.com/pymc-devs/nutpie) which is a fast NUTS sampler written in Rust. Not only is it fast, but it has an awesome progress bar.

## Run this example with marimo:

Run with following command:

```terminal
uvx marimo edit https://github.com/williambdean/pymc-marimo-example/blob/main/readme-example.py
```

## Local Development

I've built this example using [`pixi`](https://pixi.sh/latest/)

```terminal
pixi install
pixi run marimo edit readme-example.py
```
