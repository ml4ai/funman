#!/bin/bash

USER=$(id -un $UID)
jupyter notebook --allow-root --ip 0.0.0.0 --no-browser /home/$USER/funman/notebooks

