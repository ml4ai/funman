# funman

The `funman` package implements multiple simulator model analysis methods.
Current methods include:
- Parameter Synthesis
- Querying a Simulation

## **Use cases**
### **Compare Bilayer Model to Simulator**:

This use case involves the simulator and FUNMAN reasoning about the CHIME
SIR bilayer model.  See test `test_use_case_bilayer_consistency` in `test/test_use_cases.py`.

It first uses a SimulationScenario to execute the input simulator
function and evaluate the input query function using the simulation results.
In the example below this results in the run_CHIME_SIR simulator function and
evaluating whether or not the number of infected crosses the provided threshold with a custom QueryFunction referencing the `does_not_cross_threshold` function.

It then constructs an instance of the ConsistencyScenario class to evaluate whether a BilayerModel will satisfy the given query. The query asks whether the
number of infected at any time point exceeds a specified threshold.

Once each of these steps is executed the results are compared. The test will
succeed if the SimulatorScenario and ConsistencyScenario agree on the response to the query.

```python
    def compare_against_CHIME_Sim(
        self, bilayer_path, init_values, infected_threshold
    ):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)

        query = QueryLE("I", infected_threshold)

        funman = Funman()

        sim_result: SimulationScenarioResult = funman.solve(
            SimulationScenario(
                model=SimulatorModel(run_CHIME_SIR),
                query=QueryFunction(does_not_cross_threshold),
            )
        )

        consistency_result: ConsistencyScenarioResult = funman.solve(
            ConsistencyScenario(
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                ),
                query=query,
            )
        )

        # assert the both queries returned the same result
        return sim_result.query_satisfied == consistency_result.query_satisfied

    def test_use_case_bilayer_consistency(self):
        """
        This test compares a BilayerModel against a SimulatorModel to
        determine whether their response to a query is identical.
        """
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}
        assert self.compare_against_CHIME_Sim(
            bilayer_path, init_values, infected_threshold
        )
```

### **Parameter Synthesis**

See tests `test_use_case_simple_parameter_synthesis` and `test_use_case_bilayer_parameter_synthesis` in  `test/test_use_cases.py`.

The base set of types used during Parameter Synthesis include:

- a list of Parameters representing variables to be assigned
- a Model to be encoded as an SMTLib formula 
- a Scenario container representing a set of parameters and model
- a SearchConfig to configure search behavior
- the Funman interface that runs analysis using scenarios and configuration data

In the following example two parameters, x and y, are constructed. A model is 
also constructed that says 0.0 < x < 5.0 and 10.0 < y < 12.0. These parameters
and model are used to define a scenario that will use BoxSearch to synthesize
the parameters. The Funman interface and a search configuration are also 
defined. All that remains is to have Funman solve the scenario using the defined
configuration.

```python
def test_use_case_simple_parameter_synthesis(self):
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)

        formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                [
                    Parameter(name="x", symbol=x),
                    Parameter(name="y", symbol=y),
                ],
                EncodedModel(formula),
            )
        )
        assert result
```

As an additional parameter synthesis example, the following test case
demonstrates how to perform parameter synthesis for a bilayer model.  The
configuration differs from the example above by introducing bilayer-specific
constraints on the initial conditions (`init_values` assignments), parameter
bounds (`parameter_bounds` intervals) and a model query.

```python
    def test_use_case_bilayer_parameter_synthesis(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 3
        init_values = {"S": 9998, "I": 1, "R": 1}

        lb = 0.000067 * (1 - 0.5)
        ub = 0.000067 * (1 + 0.5)

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                parameters=[Parameter(name="beta", lb=lb, ub=ub)],
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                    parameter_bounds={
                        "beta": [lb, ub],
                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                    },
                ),
                query=QueryLE("I", infected_threshold),
            ),
            config=SearchConfig(tolerance=1e-8),
        )
        assert len(result.parameter_space.true_boxes) > 0 
        assert len(result.parameter_space.false_boxes) > 0 
```
---

## Development Setup

### Pre-commit hooks
FUNMAN has a set of pre-commit hooks to help with code uniformity. These hooks
will share the same rules as any automated testing so it is recommended to
install the hooks locally to ease the development process.

To use the pre-commit hooks you with need the tools listed in
`requirements-dev.txt`. These should be installed in the same environment where
you git tooling operates.
```bash
pip install -r requirements-dev.txt
```

Once install you should be able to run the following from the root of the repo:
```bash
make install-pre-commit-hooks
```

Once installed you should begin to receive isort/black feedback when
committing. These should not alter the code during a commit but instead just
fail and prevent a commit if the code does not conform to the specification.

To autoformat the entire repo you can use:
```bash
make format
```

### Code coverage
Pytest is configured to generate code coverage, and requires the `pytest-cov`
package to be installed.  The `pytest-cov` package is included in the
`requirements-dev.txt` (see above) and will be installed with the other dev
dependencies.  The code coverage will be displayed in the pytest output (i.e.,
`term`) and saved to the `coverage.xml` file.  The `Coverage Gutters` VSCode
plugin will use the `coverage.xml` to display code coverage highlighting over
the source files.

### Development Setup: Docker dev container
FUNMAN provides tooling to build a Docker image that can be used as a
development container. The image builds on either arm64 or amd64 architectures.

The dev container itself is meant to be ephemeral. The `launch-dev-container`
command will delete the existing dev container if an newer image has been made
available in the local cache. Any data that is meant to be retained from the
dev-container must be kept in one of the mounted volumes.

The dev container supports editing and rebuilding of dreal4 as well. This
requires that a dreal4 repository is cloned as a sibling to the funman
directory (../dreal4). So long as that directory is present, the next time the
funman-dev container is created will also result in a bind mount of the dreal4
directory to the container.

# Build the image:
```bash
make build
```

# Launch the dev container:
```bash
make launch-dev-container
```

# If building a local dreal4 instead of the built-in version:
```bash
# from within the container
update-dreal
```

### (DEPRECATED) Development Setup: Ubuntu 20.04
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

### (DEPRECATED) Development Setup: OSX M1

```bash
# install python 3.9
brew install python@3.9 
# install dev dependencies
brew install make
pip3 install --user pipenv
# install pygraphviz dependencies
brew install graphviz pkg-config
# install z3
brew install z3
# install miniconda
brew install miniforge
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

# Generating docs
```bash
pipenv run pip install sphinx sphinx_rtd_theme matplotlib

# Needed only if the gh-pages branch is not at origin
make init-pages 

# Run sphinx and pyreverse on source, generate docs/
make docs 

# push docs/ to origin
make deploy-pages 
```
