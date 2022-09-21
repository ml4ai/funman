# funman

## Development Setup

### Development Setup: Ubuntu 20.04
```bash
# install python 3.9
sudo apt install python3.9 python3.9-dev
# install dev dependencies
sudo apt install make
pip install --user pipenv
# install pygraphviz dependencies
sudo apt install graphviz libgraphviz-dev pkg-config
# Initialize development environment
make setup-dev-env
```

### Development Setup: OSX M1

```bash
# install python 3.9
brew install python3.9 python3.9-dev
# install dev dependencies
brew install make
pip install --user pipenv
# install pygraphviz dependencies
brew install graphviz pkg-config
# install z3
brew install z3
# install miniconda
brew install miniconda
# Initialize development environment
make setup-conda-dev-env
```

#### **Z3 issue**

On the M1, installing with conda gets pysmt with z3 for the wrong architecture. To fix this, if it happens, replace the `z3lib.dylib` in your virtual environment (in my case this was `.venv/lib/python3.9/site-packages/z3/lib/libz3.dylib`) with a symbolic link to the library that you get from your brew install of z3.  For example

    ln -s /opt/homebrew/Cellar/z3/4.11.0/lib/libz3.dylib ~/projects/askem/funman/.venv/lib/python3.9/site-packages/z3/lib/

---
#### **Pipenv issue and conda**

When I (rpg) tried to set up the environment with only pipenv (`make setup-dev-env`), it didn't work because `pip` tried to build the pygraphviz wheel and when it did, it used the Xcode compiler, which did not work with the version of graphviz I had installed with brew.

Suggest dealing with this by using `setup-CONDA-dev-env` [caps for emphasis] instead.

---
#### **Error during setup: "Could not find a version that matches"**
Try updating pipenv: `pip install pipenv --upgrade`
