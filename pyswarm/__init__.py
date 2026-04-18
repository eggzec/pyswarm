"""
===========================================
PySwarm: Particle Swarm Optimization (PSO)
===========================================

Authors: Abraham Lee
         Sebastian Castillo-Hair
         Saud Zahir

Copyright (c) 2013, Abraham D. Lee
"""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from pyswarm.pso import pso


__all__ = ["pso"]
