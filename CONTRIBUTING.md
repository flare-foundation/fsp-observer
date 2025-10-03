# CONTRIBUTING

## Project tooling

### Python environment and dependencies

```sh
python -m venv venv
source ./venv/bin/activate
pip install -U -r requirements.txt -r dev-requirements.txt
```

### Git hooks and linters

commited code should be linted and formatted
```sh
# format
ruff format
# lint
ruff check 
# optionally auto fix
ruff check --fix
```

enforce this check with a pre commit hook
```sh
pre-commit install
```
