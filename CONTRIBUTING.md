# CONTRIBUTING

## Project tooling

### Python environment and dependencies

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install the
project together with dev tooling into a local virtual environment:

```sh
uv sync
```

Run commands inside the environment with `uv run`, e.g. `uv run python main.py`.

### Git hooks and linters

commited code should be linted and formatted

```sh
# format
uv run ruff format
# lint
uv run ruff check
# optionally auto fix
uv run ruff check --fix
```

enforce this check with a pre commit hook

```sh
pre-commit install
```
