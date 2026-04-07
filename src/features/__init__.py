"""Feature pipeline registry.

All feature builders are collected here into FEATURE_BUILDERS, a dict
mapping pipeline name to its builder function. The training loop and
experiment scripts import this dict to look up the right feature
engineering for each model.
"""

from src.features.basic import BASIC_BUILDERS
from src.features.enriched import ENRICHED_BUILDERS
from src.features.origstats import ORIGSTATS_BUILDERS
from src.features.top_pipe import build_top_pipe
from src.features.pairproto import build_pairwise_proto
from src.features.stacking import STACKING_BUILDERS

FEATURE_BUILDERS = {}
FEATURE_BUILDERS.update(BASIC_BUILDERS)
FEATURE_BUILDERS.update(ENRICHED_BUILDERS)
FEATURE_BUILDERS.update(ORIGSTATS_BUILDERS)
FEATURE_BUILDERS.update(STACKING_BUILDERS)
FEATURE_BUILDERS["top_pipe"] = build_top_pipe
FEATURE_BUILDERS["pairwise_proto"] = build_pairwise_proto
