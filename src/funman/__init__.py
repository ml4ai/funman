"""
The funman package implements multiple simulator model analysis methods.  Current methods include:

- Simulation: running original simulator and querying the results.

- Parameter Synthesis: Generating feasible values for model parameters.

- Consistency: Check that a parameterized model is self-consistent.
"""


from ._version import __version__
from .funman import *
from .representation import *
from .model import *
from .scenario import *
from .utils import *
