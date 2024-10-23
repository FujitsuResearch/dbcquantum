#!/bin/sh -xe

# poetry update

poetry run isort src
poetry run black src
poetry run black notebook
poetry run pyright src
poetry run flake8 src/test --max-line-length=130
poetry run pytest src/test/

rm dist -rf
poetry build
# poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev,jupyter
poetry export -f requirements.txt --output requirements-jupyter.txt --without-hashes --with jupyter

rm doc/source/apidoc/* -rf 
poetry run sphinx-apidoc -f -e -o doc/source/apidoc src/dbcquantum --module-first
(cd doc && poetry run make html)
{ set +x; } 2>/dev/null
read -p "Do you want to run http.server? [y/N]: " ANS
case $ANS in
    [Yy]* )
        set -x
        poetry run python -m http.server --directory doc/build/html
        ;;
    * )
        ;;
esac