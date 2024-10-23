# Design by Contract Framework for Quantum Software (DbCQuantum)

This framework enables us to verify quantum applications through assertions. We can write any assertions about the pre-states and post-states of quantum circuits. These assertions are verified on a quantum computer simulator using test cases. Additionally, the framework allows us to verify the post-processing of measurement results.


## Installation

Install python 3.11 via [pyenv](https://github.com/pyenv/pyenv) or [directly](https://www.python.org/) and run the following at the root of this repository.

```sh
python -m venv .venv       # recommended
source .venv/bin/activate  # recommended
pip install --upgrade pip # recommended
pip install dist/dbcquantum-*.whl
```

If you want to use jupyter notebook example:

```sh
pip install -r requirements-jupyter.txt
```

## Usage

You can see example codes in the `example` directory.

An example of a success pattern.

```sh
python example/hadamard_test/success.py 
```

An example of a failed pattern.

```sh
python example/hadamard_test/fail.py 
```

The API references are available at
https://fujitsuresearch.github.io/dbcquantum/

## Examples of Our Paper

Please refer to the `notebook/paper` directory.
You can run them by the following command:

```sh
python -m venv .venv       # recommended
source .venv/bin/activate  # recommended
pip install --upgrade pip # recommended
pip install dist/dbcquantum-*.whl
pip install -r requirements-jupyter.txt
jupyter notebook
```

## Development

Our project is managed by [poetry](https://github.com/python-poetry/poetry) (version 1.8.2).  
You can install all dependencies via poetry.  

- via poetry:

    ```sh
    python -m venv .venv       # recommended
    poetry config --local virtualenvs.in-project true # recommended
    poetry install
    ```

When you update something, run the following code and solve all errors.

```sh
chmod u+x run_before_push.sh 
./run_before_push.sh
```
