"""One hash-authorized original TabM fit; Torch-only, never reads held outcomes."""
from model import HERE, ROUND
from model_embedded import *
import sys
import argparse
import hashlib
import json
import time
import resource
import numpy as np

sys.path.insert(0, str(ROUND / 'queue_data'))
from common import CACHE, ME, FIT_END, sha, dump, require_release, feature_names


def train(arm, seed):
    assert arm in ['T0', 'TE'] and seed in range(3)
    assert 'lightgbm' not in sys.modules
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    release, manifest = require_release([arm, seed])
    cfg = json.loads((HERE / 'TRAIN_CONFIG.json').read_text())
    assert release['tabm_config_sha256'] == sha(HERE / 'TRAIN_CONFIG.json')
    assert torch.backends.mps.is_available() and cfg['device'] == 'mps'
    assert cfg['epochs'] == 8 and cfg['batch'] == 1024
    for path, expected in cfg['source_sha256'].items():
        assert sha(path) == expected, path
    sem = arm == 'TE'
    suffix = 'semantic210' if sem else 'base196'
    spec_path = ME / f'TRANSFORM_{suffix}.json'
    spec = json.loads(spec_path.read_text())
    path = CACHE / f'fit_matched_{suffix}.npy'
    assert sha(path) == spec['tables'][str(path)]['sha256']
    if sem:
        assert json.loads((ME / 'OOF_ASSEMBLED.json').read_text())['no_self_fit_encoder_features']
    target = HERE / 'models' / f'{arm}_seed{seed}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    x = np.load(path, mmap_mode='r')
    with np.load(CACHE / 'fit_meta.npz') as md:
        labels = md['label'].copy()
        original_weight = md['weight'].copy()
    features = feature_names(sem)
    assert x.shape == (manifest['fit_rows'], len(features))
    assert np.isfinite(labels).all() and np.isin(labels, [0., 1.]).all()
    assert original_weight.dtype == np.float64 and abs(original_weight.mean() - 1.) < 1e-12
    weights = original_weight.astype(np.float32)
    weight_cast_error = float(np.max(np.abs(weights.astype(np.float64) - original_weight)))
    actual_seed = cfg['seed_base'] + seed
    torch.manual_seed(actual_seed)
    torch.mps.manual_seed(actual_seed)
    model = make_model(len(features))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.to('mps').train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(actual_seed)
    start = time.time()
    logs = []
    optimizer_steps = 0
    for epoch in range(cfg['epochs']):
        order = rng.permutation(len(x))
        total = 0.
        steps = 0
        for begin in range(0, len(order), cfg['batch']):
            ids = order[begin:begin + cfg['batch']]
            values = np.asarray(x[ids], dtype=np.float32)
            assert np.isfinite(values).all()
            a = torch.from_numpy(values).to('mps')
            y = torch.from_numpy(labels[ids]).to('mps')
            w = torch.from_numpy(weights[ids]).to('mps')
            opt.zero_grad(set_to_none=True)
            logits = model(a)
            loss = weighted_member_bce(logits, y, w)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip_norm'])
            assert torch.isfinite(norm)
            opt.step()
            total += float(loss.detach().cpu()) * len(ids)
            steps += 1
            optimizer_steps += 1
            if steps % 500 == 0:
                print('R20_TABM_STEP', arm, seed, epoch + 1, steps,
                      round(time.time() - start, 1), flush=True)
        rec = dict(epoch=epoch + 1, steps=steps, weighted_training_BCE=total / len(x),
                   seconds=time.time() - start)
        logs.append(rec)
        print('R20_TABM_EPOCH', arm, seed, json.dumps(rec), flush=True)
    torch.mps.synchronize()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    updated = {k: not torch.equal(initial[k], value) for k, value in state.items()}
    assert all(updated.values()) and all(torch.isfinite(v).all() for v in state.values())
    model.eval()
    with torch.no_grad():
        # Fit-row numerical check only; no held labels, scores, or prediction selection.
        probe = probabilities(model(torch.from_numpy(np.asarray(x[:128])).to('mps')))
        assert torch.isfinite(probe).all() and ((probe >= 0) & (probe <= 1)).all()
    torch.save(dict(state_dict=state, architecture=CONFIG, features=features, arm=arm, seed=actual_seed,
                    fit_end=FIT_END, transform_sha256=sha(spec_path),
                    input_manifest_sha256=sha(ME / 'INPUT_MANIFEST.json'),
                    train_config_sha256=sha(HERE / 'TRAIN_CONFIG.json')), target)
    info = dict(PASS=True, arm=arm, seed_slot=seed, seed=actual_seed, fit_end=FIT_END,
                rows=len(x), features=features, epochs=cfg['epochs'], batch=cfg['batch'],
                architecture=CONFIG, parameters=sum(p.numel() for p in model.parameters()),
                optimizer_steps=optimizer_steps, updated_parameter_tensors=updated,
                finite_model_and_fit_prediction=True, original_weight_dtype='float64',
                device_weight_dtype='float32', maximum_weight_cast_error=weight_cast_error,
                model_sha256=sha(target), transform_sha256=sha(spec_path),
                input_manifest_sha256=sha(ME / 'INPUT_MANIFEST.json'),
                train_config_sha256=sha(HERE / 'TRAIN_CONFIG.json'),
                seconds=time.time() - start, peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,
                MPS_current_allocated_GB=torch.mps.current_allocated_memory()/1e9,
                logs=logs, no_held_outcomes_read=True)
    dump(target.with_suffix('.json'), info)
    x._mmap.close()
    print('R20_TABM_DONE', arm, seed, info['seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['T0', 'TE'])
    p.add_argument('seed', type=int, choices=range(3))
    a = p.parse_args()
    train(a.arm, a.seed)
