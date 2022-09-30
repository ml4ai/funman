"""
The funman package implements multiple simulator model analysis methods.  Current methods include:

- Simulation: running original simulator and querying the results.

- Parameter Synthesis: Generating feasible values for model parameters.
"""

import funman.funman
from funman._version import __version__

def main():
    return funman.funman.Funman()

if __name__ == "main":
    main()
    