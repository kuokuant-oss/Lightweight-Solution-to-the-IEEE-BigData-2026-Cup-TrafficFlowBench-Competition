"""Load the FE models fitted in s6 (s6_FE_models.pkl) and score rows (per-window mean softmax over 24 configs)."""
import numpy as np, pandas as pd
import common as C
MODELS = pd.read_pickle(C.ME / 's6_FE_models.pkl')


def softmax_by(key, s):
    s = pd.Series(s); m = s.groupby(key.to_numpy()).transform('max').to_numpy(); e = np.exp(s.to_numpy() - m)
    return e / pd.Series(e).groupby(key.to_numpy()).transform('sum').to_numpy()


def fe_score(rows):
    acc = np.zeros(len(rows)); ok = rows.panel.isin(C.MS + ['D7_I210_W']).to_numpy()
    for key, mods in MODELS.items():
        s = np.full(len(rows), -50.0)
        for p, m in mods.items():
            ix = (rows.panel == p).to_numpy()
            if ix.any(): s[ix] = m.score(rows[ix])
        acc += softmax_by(rows.key, s)
    acc /= len(MODELS); acc[~ok] = np.nan; return acc
