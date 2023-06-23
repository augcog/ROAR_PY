# ROAR_PY

Supports python >= 3.8, carla >= 0.9.12

## Documentation

Please go to our [Documentation Website](https://roar.gitbook.io/roar_py-documentation/) for more info

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Code Structures

| Folder | Description |
| --- | --- |
| `roar_py_interface` | Contains all the interfaces that interacts with a simulator / real world. |
| `roar_py_carla_implementation` | Contains all the implementations of the interfaces for Carla. |
| `roar_py_remote` | Contains all interface / wrappers for supporting remote + shared ownership control. |

## Examples & Documents

Please refer to `examples/` folder for examples and `docs/` folder for documents.

## Unit Tests

All tests are located under `tests/` folder. To run all tests, simply run:

```bash
pytest tests
```

## Contributing

We accept contributions via [Pull Requests on Github](https://github.com/FHL-VIVE-Center-for-Enhanced-Reality/ROAR_PY/pulls). We do manual code review and testing before merging any of the PRs. Before submitting a PR, please make sure that your code passes all the tests and linters.