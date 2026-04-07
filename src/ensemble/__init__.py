"""Ensemble blending library.

Provides reusable blending techniques (blend.py) and declarative
ensemble definitions (definitions.py). The experiment script
builds ensembles by reading definitions and applying techniques.
"""

from src.ensemble.blend import rank_blend, hillclimb, band_gate_blend
from src.ensemble.definitions import ENSEMBLE_DEFS
