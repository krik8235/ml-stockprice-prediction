# Delta Lake Lakehouse Architecture with Spark and Airflow DAGs


## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Quick Start](#quick-start)
- [Package Management](#package-management)
- [Running local](#running-local)
- [Pre-commit hooks](#pre-commit-hooks)
- [Testing](#testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv run spark-submit --packages io.delta:delta-spark_2.13:4.0.0,org.apache.hadoop:hadoop-aws:3.4.0,com.amazonaws:aws-java-sdk-bundle:1.12.262 src/main.py {TICKER} --cache-clear
```

(Replace *{TICKER}* with a ticker of your choice.)


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


## Running local

```bash
EXPORT RUST_LOG="info,datafusion_datasource_parquet=error"
uv run src/main.py --cache-clear
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
