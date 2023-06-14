#!/bin/bash

pip install datamodel-code-generator || echo "Didn't install datamodel-code-generator (already installed?)"
datamodel-codegen --input ~/Model-Representations/petrinet/petrinet_schema.json --input-file-type jsonschema --output src/funman/model/generated-models/petrinet.py
datamodel-codegen --input ~/Model-Representations/regnet/regnet_schema.json --input-file-type jsonschema --output src/funman/model/generated-models/regnet.py