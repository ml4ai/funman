#!/usr/bin/env bash

set -e

# Change directory to scripts directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR

# Setup environment
SERVER_URL=http://127.0.0.1:8190
EXAMPLE_PATH=../resources/terarium-tests.json

if [[ ( $@ == "--help") ||  $@ == "-h"  || $# -ne 1 ]]
then 
	echo "Usage: $0 <example>"
    ./scripts/terarium_examples.py $SERVER_URL $EXAMPLE_PATH
	exit 0
fi
./scripts/terarium_examples.py $SERVER_URL $EXAMPLE_PATH -r $1

# Return to original directory
popd