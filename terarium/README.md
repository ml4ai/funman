# Funman API Example Queries

## Notes about the examples environment
These examples try to minimize the required environment for local testing.

The following is required and assumed present within your environment:
- docker
- python3
- The python 'requests' package

NOTE: A `requirements.txt` file is provided such that one can install required
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
The `scripts` directory contains utility scripts.
The `examples` directory contains specific example scripts.

These examples internally depends on:
- `./scripts/post-query.py` for making POST requests to `/api/queries`
- `./scripts/get-status.py` for making GET requests to `/api/queries/{uuid}`

These are basic helper scripts for making requests and printing some info.

Running an example:
```
./examples/default_example.sh
```

This will output the following:
```
Using default request:
The POST payload: 
{
    "model": <Contents of ../resources/amr/petrinet/mira/models/scenario1_a.json>
    "request": {}
}
Query received work id: 89cd294a-0609-45d3-8bea-85f1ad575426
Getting status for query: 89cd294a-0609-45d3-8bea-85f1ad575426
{
  "id": "89cd294a-0609-45d3-8bea-85f1ad575426",
  "model": "Removed by get-status.py for brevity",
  "request": "Removed by get-status.py for brevity",
  "done": true,
  "parameter_space": {
    "num_dimensions": 12,
    "true_points": [
      {
        "label": "true",
        "values": {
          "num_steps": 0.0,
          "step_size": 1.0
        }
      }
    ]
  }
}
```