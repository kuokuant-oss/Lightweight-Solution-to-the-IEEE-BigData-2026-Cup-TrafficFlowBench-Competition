"""Records model fitting and loaded code in every Python process of an audited Task 2 run.

Imported at interpreter start (from the run tree's copy of sitecustomize.py, see audit/run_no_refit.sh)
when TFB_AUDIT_PROBE names a log file. Appends JSON lines to that file:

- {"kind": "fit", "call": ..., "program": ...} each time a model-fitting entry point is called:
  lightgbm train / cv / Booster(train_set=...) / Booster.update, every scikit-learn estimator's fit,
  fit_transform and partial_fit, catboost fit, scipy.optimize minimize / least_squares / nnls /
  lsq_linear / curve_fit / minimize_scalar / root, numpy polyfit / lstsq, torch Optimizer.step;
- {"kind": "exit", "program": ..., "modules": [...]} at exit: every loaded module file under the run
  tree's trafficflowbench/ (the code the program actually ran), and the program itself.

The wrappers only record and then call the original function, so outputs are unchanged.
"""
import atexit, functools, importlib.abc, json, os, sys

LOG = os.environ.get('TFB_AUDIT_PROBE')
REPO = os.environ.get('TFB_REPO_ROOT', '')
# module -> names to wrap: 'func' or 'Class.method'
TARGETS = {
    'lightgbm.engine': ['train', 'cv'],
    'lightgbm.basic': ['Booster.__init__', 'Booster.update'],
    'lightgbm.sklearn': ['LGBMModel.fit'],
    'catboost.core': ['CatBoost.fit', 'CatBoost._fit'],
    'scipy.optimize._minimize': ['minimize', 'minimize_scalar'],
    'scipy.optimize._lsq.least_squares': ['least_squares'],
    'scipy.optimize._nnls': ['nnls'],
    'scipy.optimize._lsq.lsq_linear': ['lsq_linear'],
    'scipy.optimize._minpack_py': ['curve_fit'],
    'scipy.optimize._root': ['root'],
    'numpy.lib._polynomial_impl': ['polyfit'],
    'numpy.linalg._linalg': ['lstsq'],
    'torch.optim.optimizer': ['Optimizer.step'],
}


PATCHED = []


def program():
    """Returns the program this interpreter runs: the script path relative to the run tree, or -c."""
    p = sys.argv[0] if sys.argv else ''
    if p.endswith('.py'):
        r = os.path.realpath(p)
        return os.path.relpath(r, os.path.realpath(REPO)) if REPO and r.startswith(os.path.realpath(REPO) + os.sep) else r
    return p


def record(**kw):
    """Appends one JSON line to the probe log."""
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(dict(pid=os.getpid(), program=program(), argv=sys.argv[1:6], **kw)) + '\n')


def wrap(owner, attr, label):
    """Replaces owner.attr with a wrapper that records the call (Booster.__init__ only with a train_set)."""
    orig = getattr(owner, attr, None)
    if orig is None or getattr(orig, '_tfb_probe', False):
        return
    @functools.wraps(orig)
    def w(*a, **k):
        if attr != '__init__' or k.get('train_set') is not None:
            record(kind='fit', call=label)
        return orig(*a, **k)
    w._tfb_probe = True
    setattr(owner, attr, w)
    PATCHED.append(label)


def patch(mod):
    """Wraps the entry points of a just-loaded module."""
    name = mod.__name__
    for t in TARGETS.get(name, []):
        if '.' in t:
            cls, meth = t.split('.', 1)
            if hasattr(mod, cls):
                wrap(getattr(mod, cls), meth, f'{name}.{t}')
        else:
            wrap(mod, t, f'{name}.{t}')
    if name.startswith('sklearn.'):
        # Every estimator class defined in this module: its own fit / fit_transform / partial_fit.
        for v in list(vars(mod).values()):
            if isinstance(v, type) and v.__module__ == name:
                for meth in ('fit', 'fit_transform', 'partial_fit'):
                    if meth in vars(v):
                        wrap(v, meth, f'{name}.{v.__name__}.{meth}')


class Finder(importlib.abc.MetaPathFinder):
    """Finds modules with the other finders and patches the ones of interest right after they execute."""
    def find_spec(self, fullname, path, target=None):
        if not (fullname in TARGETS or fullname.startswith('sklearn.')):
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, 'find_spec'):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None and hasattr(spec.loader, 'exec_module'):
                loader = spec.loader
                class L(importlib.abc.Loader):
                    def create_module(self, s):
                        return loader.create_module(s)
                    def exec_module(self, m):
                        loader.exec_module(m); patch(m)
                spec.loader = L()
                return spec
        return None


def at_exit():
    """Records the run-tree modules this process loaded."""
    files = set()
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and REPO and os.path.realpath(f).startswith(os.path.realpath(REPO) + os.sep):
            files.add(os.path.relpath(os.path.realpath(f), os.path.realpath(REPO)))
    p = program()
    if p.endswith('.py') and not os.path.isabs(p):
        files.add(p)
    # patched: the entry points this process had loaded and wrapped (the probe's own coverage).
    record(kind='exit', modules=sorted(files), patched=sorted(set(PATCHED)))


if LOG:
    sys.meta_path.insert(0, Finder())
    for _m in list(sys.modules.values()):   # anything already imported
        if getattr(_m, '__name__', None) in TARGETS or getattr(_m, '__name__', '').startswith('sklearn.'):
            patch(_m)
    atexit.register(at_exit)
