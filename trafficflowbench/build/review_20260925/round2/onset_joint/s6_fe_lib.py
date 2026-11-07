"""Load the FE models fitted in s6 (s6_FE_models.pkl) and score rows (per-window mean softmax over 24 configs)."""
# FE: the site-choice ensemble fitted in onset_joint step s6 (its fitting
# script is not shipped). MODELS maps each of the 24 configurations to
# {panel: fitted conditional-logit model with a score(rows) method}.
# synthesis/sv2_official_FEA1.py imports fe_score to choose the queuing site
# of the official onset windows on the multi-site panels.
import numpy as np, pandas as pd
import common as C
MODELS = pd.read_pickle(C.ME / 's6_FE_models.pkl')


def softmax_by(key, s):
    """Returns the softmax of scores `s` within each group of `key` (one window per group)."""
    s = pd.Series(s); m = s.groupby(key.to_numpy()).transform('max').to_numpy(); e = np.exp(s.to_numpy() - m)
    return e / pd.Series(e).groupby(key.to_numpy()).transform('sum').to_numpy()


def fe_score(rows):
    """Returns the FE score of each per-site row.

    The score is the mean over the configurations of the within-window
    softmax of the model scores. Rows of a panel without a model in a
    configuration get the constant score -50 there. Rows outside the three
    multi-site panels and D7_I210_W get NaN.

    Args:
      rows: per (window key, site) rows with panel and the model features.
    """
    acc = np.zeros(len(rows)); ok = rows.panel.isin(C.MS + ['D7_I210_W']).to_numpy()
    for key, mods in MODELS.items():
        s = np.full(len(rows), -50.0)
        for p, m in mods.items():
            ix = (rows.panel == p).to_numpy()
            if ix.any(): s[ix] = m.score(rows[ix])
        acc += softmax_by(rows.key, s)
    acc /= len(MODELS); acc[~ok] = np.nan; return acc
