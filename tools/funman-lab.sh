#!/bin/bash

USER=$(id -un $UID)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser /home/$USER/funman/notebooks

