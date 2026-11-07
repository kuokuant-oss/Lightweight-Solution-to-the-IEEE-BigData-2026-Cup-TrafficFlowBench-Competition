"""Original-author TabM0.0.3; no local reimplementation or model fitting."""
import os
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
sys.path.insert(0, str(ROUND / 'tabm_runtime'))
import torch
import tabm

CONFIG = dict(d_out=1, n_blocks=2, d_block=128, dropout=0.1, k=32,
              arch_type='tabm', num_embeddings=None)


def make_model(features):
    assert tabm.__version__ == '0.0.3'
    return tabm.TabM.make(n_num_features=features, **CONFIG)


def weighted_member_bce(logits, target, weight):
    """Optimize each member's BCE; average losses rather than predictions."""
    assert logits.ndim == 3 and logits.shape[1:] == (32, 1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :, 0], target[:, None].expand(-1, 32), reduction='none')
    return (losses.mean(dim=1) * weight).mean()


def probabilities(logits):
    assert logits.ndim == 3 and logits.shape[1:] == (32, 1)
    return torch.sigmoid(logits[:, :, 0]).mean(dim=1)
