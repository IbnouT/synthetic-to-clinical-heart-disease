"""Ensemble configuration definitions.

Each ensemble is a declarative spec: which technique to use, which
inputs to combine, and what parameters to pass. The experiment script
reads these definitions and builds them in dependency order.

Adding a new ensemble: define it here and run the build script.
No hardcoded layer logic needed.
"""

# ---------------------------------------------------------------------------
# Ensemble specs
#
# Each entry maps an ensemble name to its configuration:
#   method      : "rank_blend", "weighted_rank_blend", "hillclimb", "band_gate"
#   inputs      : list of model/ensemble names (resolved in dependency order)
#   weights     : per-input weights (for weighted_rank_blend)
#   normalize   : whether to min-max normalize the blend output
#   params      : method-specific parameters (band gate lo/hi/weight, etc.)
#   expected_auc: known AUC for verification (optional)
# ---------------------------------------------------------------------------

ENSEMBLE_DEFS = {
    "40_cb_lr_stack_blend": {
        "method": "rank_blend",
        "inputs": [
            "40_cb_lr_stack_s42",
            "40_cb_lr_stack_s123",
            "40_cb_lr_stack_s456",
            "40_cb_lr_stack_s789",
            "40_cb_lr_stack_s2024",
        ],
        "normalize": True,
        "expected_auc": 0.955707630377204,
    },

    "21_multiseed_blend": {
        "method": "rank_blend",
        "inputs": [
            "18_cb_freq_multiseed_avg",
            "19_lr_onehot_multiseed_avg",
            "20_cb_origstats_multiseed_avg",
        ],
        "normalize": True,
        "expected_auc": 0.955692393210679,
    },

    "hill_climb_v3": {
        "method": "rank_blend",
        "inputs": [
            "40_cb_lr_stack_blend",
            "te_cb_a10",
        ],
        "normalize": False,
        "expected_auc": 0.955750623958581,
    },

    "hillclimb_v4": {
        "method": "rank_blend",
        "inputs": [
            "40_cb_lr_stack_blend",
            "te_cb_a10",
            "cb_shrink_0.01",
            "40_cb_lr_stack_s2024",
            "21_multiseed_blend",
        ],
        "normalize": False,
        "expected_auc": 0.955759453116718,
    },

    "3way_v4v3_tp10": {
        "method": "weighted_rank_blend",
        "inputs": ["hillclimb_v4", "hill_climb_v3", "top_pipe_f10_s42"],
        "weights": [0.40, 0.40, 0.20],
        "normalize": False,
        "expected_auc": 0.955883734248581,
    },

    "rankgate_narrow_w5": {
        "method": "band_gate",
        "inputs": ["3way_v4v3_tp10", "cb_pairproto_s42"],
        "params": {"weight": 0.05, "lo": 0.08, "hi": 0.32},
        "normalize_anchor": True,
    },
}
