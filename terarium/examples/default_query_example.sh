#!/usr/bin/env bash

set -e

# Change directory to scripts directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR/..

# Setup environment
SERVER_URL=http://127.0.0.1:8190
MODEL_PATH=../resources/amr/petrinet/mira/models/scenario1_a.json

# Make a POST request to $SERVER_URL/api/queries where:
# - model is the content of the file at $MODEL_PATH
# - request is unset and post-query assigns it to an empty dict
# NOTE: "post-query.py" prints the id from the response json.
# Store the printed work id into FUNMAN_WORK_ID
FUNMAN_WORK_ID=$(./scripts/post-query.py $SERVER_URL $MODEL_PATH)
# Wait for some amount of time to allow processing to execute
sleep 5
# Make a GET request to $SERVER_URL/api/queries/$FUNMAN_WORK_ID
# "get-status.py" pretty prints the response json 
./scripts/get-status.py $SERVER_URL $FUNMAN_WORK_ID

# Return to original directory
popd