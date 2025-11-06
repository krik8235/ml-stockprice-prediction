# ML Real-Time Stock Forecasting on Delta Lake Lakehouse


## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Quick Start](#quick-start)
- [Run batch learning](#run-batch-learning)
- [Tracking model versions with MLFlow](#tracking-model-versions-with-mlflow)
- [Scheduled Run with Airflow](#scheduled-run-with-airflow)
- [Package Management](#package-management)
- [Pre-commit hooks](#pre-commit-hooks)
- [Testing](#testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Quick Start

- Run the websocket server:

```bash
uv run src/websocket.py
```

The websocket server is available at **ws://localhost:8080**.

- Connect clients:

```bash
uv run src/client.py
```

<hr />


## Run batch learning

```bash
EXPORT RUST_LOG="info,datafusion_datasource_parquet=error"
uv run src/model/tuning.py --ticker <TICKER> --r=<R>
```

Replace <TICKER> and <R> with a ticker and budget (total epochs for hyperband tuning) of your choice. Default: TICKER=NVDA, R=100

<hr />


## Tracking model versions with MLFlow

```bash
mlflow ui --port 5000
```

Logs are available at **http://127.0.0.1:5000**.

<hr />

## Scheduled Run with Airflow

```bash
airflow api-server -p 8000
```

Airflow DAGs are available at **http://127.0.0.1:8000**.

<hr />

## Package Management

- Add a package: `uv add <package>`
- Remove a package: `uv remove <package>`
- Run a command in the virtual environment: `uv run <command>`
- To completely refresh the environement, run the following commands:

```bash
rm -rf .venv
rm -rf uv.lock
uv cache clean
uv venv
source .venv/bin/activate
uv sync
```

<hr />


## Pre-commit hooks

Pre-commit hooks runs hooks defined in the `pre-commit-config.yaml` file before every commit.

To activate the hooks:

1. Install pre-commit hooks:

```bash
uv run pre-commit install
```

2. Run pre-commit checks manually:

```bash
uv run pre-commit run --all-files
```

Pre-commit hooks help maintain code quality by running checks for formatting, linting, and other issues before each commit.

* To skip pre-commit hooks

```bash
git commit --no-verify -m "your-commit-message"
```


## Testing

```bash
uv run pytest --cache-clear
```
