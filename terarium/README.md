# Funman API Example Queries

## Notes about the examples environment
These examples try to minimize the required environment for local testing.

The following is required and assumed present within your environment:
- docker
- python3
- The python 'requests' package

> NOTE: A `requirements.txt` file is provided such that one can install required
python packages with:
```
pip install -r requirements.txt
```
You are also free to setup your own virtual environment.

---
# General Overview
You will need two terminals:
- One to run the API server within a docker container
- One to run the example scripts

## **First** start the the API server
The `./scripts/run-api-in-docker.sh` script offers a simple method of launching a
local container running the funman API.

> NOTE: This script is meant to stay in sync with the releases (the image tag launched in the script should be 1.6.0 for release 1.6.0)

Running from the directory this README is in should look like this:
```bash
./scripts/run-api-in-docker.sh
```
Once the image is pulled the server should start with this output:
```
# INFO:     Started server process [1]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8190 (Press CTRL+C to quit)
```
The server should now be exposed at `127.0.0.1:8190`

## **Second** run one of the examples
The `run_example.sh` script in the same directory as this README
is the entry point for an examples.

You can list examples with:
```
./run_example.sh --help
```
It should print out the usage response:
```
Usage: ./terarium/run_example.sh <example>
Available examples:
-  petrinet-query-default
-  petrinet-query-with-parameters
```

Run an example with:
```
./run_example.sh petrinet-query-with-parameters
```

This will output the following:
```
Running example 'petrinet-query-with-parameters':
{
  "name": "petrinet-query-with-parameters",
  "model-path": "amr/petrinet/mira/models/scenario2_a_beta_scale_static.json",
  "request-path": "amr/petrinet/mira/requests/request2_b_default_w_compartmental_constrs.json"
}
################################################################################
################################################################################
Making POST request to http://127.0.0.1:8190/api/queries with contents:
{
    "model": <Contents of /home/jladwig/Documents/workspaces/sift/ASKEM/code/funman/resources/amr/petrinet/mira/models/scenario2_a_beta_scale_static.json>
    "request": <Contents of /home/jladwig/Documents/workspaces/sift/ASKEM/code/funman/resources/amr/petrinet/mira/requests/request2_b_default_w_compartmental_constrs.json>
}
################################################################################
################################################################################
Response for query:
{
  "id": "d713f2e0-335b-481c-8626-e087b0b91e11",
  "model": "Removed for brevity",
  "request": "Removed for brevity"
}
################################################################################
################################################################################
Work Id is 'd713f2e0-335b-481c-8626-e087b0b91e11'
Waiting for 5 seconds...
################################################################################
################################################################################
Making GET request to http://127.0.0.1:8190/api/queries/d713f2e0-335b-481c-8626-e087b0b91e11:
{
  "id": "d713f2e0-335b-481c-8626-e087b0b91e11",
  "model": "Removed for brevity",
  "request": "Removed for brevity",
  "done": true,
  "parameter_space": {
    "num_dimensions": 17
  }
}
################################################################################
################################################################################
The resulting ParameterSpace is:
{
  "num_dimensions": 17
}
~/Documents/workspaces/sift/ASKEM/code/funman
```