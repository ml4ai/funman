"""
This module contains class definitions used to represent and interact with
models in FUNMAN.  The submodules include:

* model: abstract classes for representing models

* query: classes for representing model queries

* bilayer: classes for bilayer models

* gromet: classes for GroMet models

* encoded: classes for models (hand) encoded directly as SMT formulas
"""
from .model import *
from .query import *
from .bilayer import *
from .encoded import *
from .gromet import *
