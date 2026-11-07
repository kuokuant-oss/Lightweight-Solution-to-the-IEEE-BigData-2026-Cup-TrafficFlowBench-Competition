"""Explicit, process-local relocation of frozen project Python sources.

Only project-owned modules are adapted. Files on disk, JSON provenance, source
hashes and third-party modules remain unchanged. Use bin/tfb-python to enable.
"""
# In this package the launcher is bin/tfb-run: it puts scripts/ssd_bootstrap and scripts/ on
# PYTHONPATH, sets TFB_SSD_COMPAT=1 (so ssd_bootstrap/sitecustomize.py calls install()) and runs
# main() with the program and its arguments.
# Mechanism: project-owned Python sources (see owned()) are compiled through _compile(), which
# rewrites string constants that contain an old project root and makes `from pathlib import Path`
# import the remapping Path class below; their open, compile and exec builtins remap paths too.
# The path map itself is portable_paths.py.
import ast
import builtins
import hashlib
import importlib.machinery
import inspect
import io
import json
import os
from pathlib import Path as NativePath
import re
import runpy
import sys
import types
import __future__

# REPO: this copy's trafficflowbench/ folder; ROOT: the run tree root.
REPO = NativePath(__file__).resolve().parents[1]
ROOT = REPO.parent
# Hashes of this adapter and of the path map; they go into each audit record (see _compile).
ADAPTER_SHA256 = hashlib.sha256(NativePath(__file__).read_bytes()).hexdigest()
PORTABLE_BINDING = {}
for _rel in ('scripts/portable_paths.py', 'environment/portable-files.json'):
    _p = REPO / _rel
    if _p.is_file():
        PORTABLE_BINDING[_rel] = hashlib.sha256(_p.read_bytes()).hexdigest()
from portable_paths import OLD_ROOTS, resolve_string, rebase_text as portable_rebase_text
_PATTERNS = [re.compile(re.escape(root) + r'(?=$|[/\s\x22\x27])') for root in OLD_ROOTS]
# The concrete pathlib class of this OS (PosixPath on macOS and Linux), base of the remapping Path.
_BASE_PATH = type(NativePath())
# Unpatched builtins, used by the adapter itself.
_NATIVE_COMPILE, _NATIVE_EXEC, _NATIVE_OPEN = builtins.compile, builtins.exec, builtins.open
# Compiler flags of every __future__ feature, so _compile inherits the caller's future imports
# the way compile() does.
_FUTURE_FLAGS = sum(getattr(__future__, n).compiler_flag for n in __future__.all_feature_names)
_INSTALLED = False
# Audit records of every source that the transform changed (see _compile).
TRANSFORMS = []


def rebase_text(value):
    """Rewrites old project roots in source text (portable_paths.rebase_text)."""
    return portable_rebase_text(value)


def resolve(value):
    """Maps a path from an old root onto the run tree (portable_paths.resolve_string)."""
    return resolve_string(value)


def canonical(value):
    """Stable path identity for pre-migration legacy checkpoint digests only."""
    # Inverse direction: run-tree paths go back to OLD_ROOTS[0], so digests match the old ones.
    if isinstance(value, str):
        here = str(ROOT)
        if value == here or value.startswith(here + '/'):
            return OLD_ROOTS[0] + value[len(here):].replace('\\', '/')
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            mapped = canonical(key)
            if mapped in result:
                raise ValueError('Ambiguous old/new path keys in legacy digest')
            result[mapped] = canonical(item)
        return result
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, tuple):
        return tuple(canonical(v) for v in value)
    return value


class Path(_BASE_PATH):
    """Normal Path semantics with only the two obsolete project roots remapped."""
    # Every constructor and join returns this class, so paths built from an old root stay mapped.
    def __new__(cls, *args, **kwargs):
        """Creates the path at its remapped location."""
        return super().__new__(cls, resolve(NativePath(*args)), **kwargs)

    def __init__(self, *args, **kwargs):
        """Initialises the path (from Python 3.12 pathlib takes the arguments in __init__)."""
        if sys.version_info >= (3, 12):
            super().__init__(resolve(NativePath(*args)), **kwargs)

    def __truediv__(self, other):
        """Joins like pathlib and returns the remapping Path type."""
        return Path(str(super().__truediv__(other)))

    def joinpath(self, *other):
        """Joins like pathlib and returns the remapping Path type."""
        return Path(str(super().joinpath(*other)))

    def with_name(self, name):
        """Like pathlib, returning the remapping Path type."""
        return Path(str(super().with_name(name)))

    def with_suffix(self, suffix):
        """Like pathlib, returning the remapping Path type."""
        return Path(str(super().with_suffix(suffix)))

    def read_text(self, *args, **kwargs):
        """Reads the text; project .py sources come back with old roots rebased."""
        text = super().read_text(*args, **kwargs)
        # Legacy generators read and compile Python source fragments. Keep binary
        # reads untouched so sha256 and source manifests still see original bytes.
        return rebase_text(text) if self.suffix == '.py' and owned(self) else text


def owned(filename):
    """Tells whether a file is a project Python source that the path layer adapts.

    Args:
        filename: path in any form accepted by resolve().

    Returns:
        True for .py files under the run tree root, except the path-layer files themselves and
        files under .git, virtual environments, site-packages, __pycache__, runtime,
        ssd_bootstrap or external.
    """
    try:
        p = NativePath(resolve(filename)).resolve()
        rel = p.relative_to(ROOT)
    except (TypeError, ValueError):
        return False
    if p in {NativePath(__file__).resolve(), REPO / 'scripts/migration_paths.py'} or (p.parent == REPO / 'scripts' and p.name.startswith(('portable_', 'setup_portable', 'windows_dispatch'))):
        return False
    excluded = {'.git', '.venv', 'venv', 'site-packages', '__pycache__', 'runtime', 'ssd_bootstrap', 'external'}
    return p.suffix == '.py' and not excluded.intersection(rel.parts)


class Relocate(ast.NodeTransformer):
    """AST transformer that rebases old roots and redirects `from pathlib import Path`.

    Attributes:
        literals: number of string constants changed.
        path_imports: number of `Path` names now imported from this module.
    """
    def __init__(self):
        """Starts both counters at zero."""
        self.literals = 0
        self.path_imports = 0

    def visit_Constant(self, node):
        """Returns the constant with old roots in a string value rebased."""
        if isinstance(node.value, str):
            new = rebase_text(node.value)
            if new != node.value:
                self.literals += 1
                return ast.copy_location(ast.Constant(value=new, kind=node.kind), node)
        return node

    def visit_ImportFrom(self, node):
        """Splits `from pathlib import Path, ...` so Path comes from `_tfb_ssd_runtime`."""
        if node.module == 'pathlib' and not node.level:
            paths = [n for n in node.names if n.name == 'Path']
            others = [n for n in node.names if n.name != 'Path']
            if paths:
                self.path_imports += len(paths)
                changed = ast.copy_location(ast.ImportFrom(module='_tfb_ssd_runtime', names=paths, level=0), node)
                return ([ast.copy_location(ast.ImportFrom(module='pathlib', names=others, level=0), node)] if others else []) + [changed]
        return node


def _compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
    """Compiles source like builtins.compile, after relocating it with Relocate.

    Requests for an AST only (ast.PyCF_ONLY_AST) are passed through unchanged. When the transform
    changes anything, an audit record (file, adapter hashes, original and adapted AST hashes,
    source SHA256) is added to TRANSFORMS and appended to $TFB_SSD_AUDIT_LOG if that is set.

    Returns:
        A code object, or an AST for PyCF_ONLY_AST requests.
    """
    if not dont_inherit:
        flags |= sys._getframe(1).f_code.co_flags & _FUTURE_FLAGS
    if flags & ast.PyCF_ONLY_AST:
        return _NATIVE_COMPILE(source, filename, mode, flags, True, optimize, **kwargs)
    tree = source if isinstance(source, ast.AST) else _NATIVE_COMPILE(source, filename, mode, flags | ast.PyCF_ONLY_AST, True, optimize, **kwargs)
    original = ast.dump(tree, include_attributes=False)
    mapper = Relocate()
    tree = ast.fix_missing_locations(mapper.visit(tree))
    if mapper.literals or mapper.path_imports:
        record = {'file': str(filename), 'adapter_sha256': ADAPTER_SHA256, 'portable_binding': PORTABLE_BINDING, 'pid': os.getpid(), 'path_literals': mapper.literals,
                  'path_imports': mapper.path_imports,
                  'original_ast_sha256': hashlib.sha256(original.encode()).hexdigest(),
                  'runtime_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()}
        try:
            p = NativePath(resolve(filename))
            if p.is_file():
                record['source_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        except (OSError, ValueError):
            pass
        TRANSFORMS.append(record)
        log = os.environ.get('TFB_SSD_AUDIT_LOG')
        if log:
            with _NATIVE_OPEN(log, 'a') as stream:
                stream.write(json.dumps(record) + '\n')
    return _NATIVE_COMPILE(tree, filename, mode, flags, True, optimize, **kwargs)


def _exec(source, globals=None, locals=None, *, closure=None):
    """Runs exec() with the project builtins; str or bytes source goes through _compile first."""
    if globals is None:
        frame = sys._getframe(1)
        globals = frame.f_globals
        if locals is None:
            locals = frame.f_locals
    globals.setdefault('__builtins__', PROJECT_BUILTINS)
    if isinstance(source, (str, bytes)):
        source = _compile(source, globals.get('__file__', '<project exec>'), 'exec')
    if closure is None:
        return _NATIVE_EXEC(source, globals, locals)
    return _NATIVE_EXEC(source, globals, locals, closure=closure)


def _open(file, *args, **kwargs):
    """Opens a file at its remapped path; arguments as builtins.open."""
    return _NATIVE_OPEN(resolve(file), *args, **kwargs)


# Builtins of project modules: compile, exec and open are the remapping versions above.
PROJECT_BUILTINS = dict(vars(builtins), compile=_compile, exec=_exec, open=_open)


def install():
    """Installs the path layer in this interpreter; later calls do nothing.

    Registers this module as `_tfb_ssd_runtime`, patches SourceFileLoader so project modules are
    compiled through _compile and run with PROJECT_BUILTINS, patches runpy for scripts started
    by multiprocessing spawn, and turns off writing of .pyc files.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    sys.modules['_tfb_ssd_runtime'] = sys.modules[__name__]
    loader = importlib.machinery.SourceFileLoader
    original_get_code, original_exec_module = loader.get_code, loader.exec_module

    def get_code(self, fullname):
        """Returns the relocated code object of a project module; others load normally."""
        if not owned(self.path):
            return original_get_code(self, fullname)
        source = NativePath(resolve(self.path)).read_bytes()
        return _compile(source, str(resolve(self.path)), 'exec', dont_inherit=True)

    def exec_module(self, module):
        """Runs a project module with PROJECT_BUILTINS; others run normally."""
        if not owned(self.path):
            return original_exec_module(self, module)
        module.__dict__['__builtins__'] = PROJECT_BUILTINS
        original_exec_module(self, module)
        # Preserve legacy resume signatures when only absolute mount roots differ.
        # No hash or artifact check is bypassed: source bytes remain the originals.
        if NativePath(resolve(self.path)) == REPO / 'build/review_20261017_r2/common.py':
            original_digest = module.digest
            module.digest = lambda value: original_digest(canonical(value))

    loader.get_code = get_code
    loader.exec_module = exec_module
    # multiprocessing.spawn executes the main script through runpy rather than a
    # SourceFileLoader. Adapt that project script and its local builtins as well.
    original_run_code = runpy._run_code
    def run_code(code, run_globals, init_globals=None, mod_name=None, mod_spec=None,
                 pkg_name=None, script_name=None):
        """Recompiles a project script through _compile before runpy runs it."""
        filename = script_name or (getattr(mod_spec, 'origin', None) if mod_spec else None)
        if filename and owned(filename):
            actual = NativePath(resolve(filename))
            code = _compile(actual.read_bytes(), str(actual), 'exec', dont_inherit=True)
            init_globals = dict(init_globals or {}, __builtins__=PROJECT_BUILTINS)
        return original_run_code(code, run_globals, init_globals, mod_name, mod_spec,
                                 pkg_name, script_name)
    runpy._run_code = run_code
    sys.dont_write_bytecode = True


def main(args=None):
    """Runs a program the way the python command does, with the path layer installed.

    Accepted forms: leading -u, -B, -O, -OO, -q, -W, -X flags (the interpreter is re-executed with
    them), no argument (interactive console), --version or -V, -c <code>, -m <module>, - (read
    stdin) and <script> [args]. Script arguments are mapped with resolve().

    Args:
        args: command-line arguments after the interpreter name; defaults to sys.argv[1:].

    Raises:
        SystemExit: a -W or -X flag has no value, or another unsupported flag is given.
    """
    args = list(sys.argv[1:] if args is None else args)
    install()
    # Re-exec the real interpreter so flags retain their normal Python semantics.
    flags = []
    while args and (args[0] in ('-u', '-B', '-O', '-OO', '-q') or args[0].startswith(('-W', '-X'))):
        flag = args.pop(0)
        flags.append(flag)
        if flag in ('-W', '-X'):
            if not args:
                raise SystemExit('Missing value for ' + flag)
            flags.append(args.pop(0))
    if flags:
        os.execv(sys.executable, [sys.executable, *flags, '-c', 'import ssd_runtime; ssd_runtime.main()', *args])
    if args and args[0] == '--':
        args.pop(0)
    if not args:
        import code
        code.interact(local={'__builtins__': PROJECT_BUILTINS})
        return
    if args[0] in ('--version', '-V'):
        print('Python ' + sys.version.split()[0])
        return
    if args[0] == '-c':
        sys.argv = ['-c', *args[2:]]
        scope = {'__name__': '__main__', '__builtins__': PROJECT_BUILTINS}
        _exec(_compile(args[1], '<command>', 'exec'), scope)
        return
    if args[0] == '-m':
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], init_globals={'__builtins__': PROJECT_BUILTINS}, run_name='__main__', alter_sys=True)
        return
    if args[0] == '-':
        sys.argv = args
        _exec(_compile(sys.stdin.read(), '<stdin>', 'exec'), {'__name__': '__main__', '__builtins__': PROJECT_BUILTINS})
        return
    if args[0].startswith('-'):
        raise SystemExit('Unsupported interpreter flag in tfb-python: ' + args[0] + '; use .venv/bin/python for interpreter-only flags')
    script = NativePath(resolve(args[0])).absolute()
    sys.argv = [str(script), *[resolve(x) for x in args[1:]]]
    sys.path.insert(0, str(script.parent))
    module = types.ModuleType('__main__')
    scope = module.__dict__
    is_project = owned(script)
    scope.update(__file__=str(script), __package__=None, __cached__=None,
                 __spec__=None, __builtins__=PROJECT_BUILTINS if is_project else vars(builtins))
    sys.modules['__main__'] = module
    compiler = _compile if is_project else _NATIVE_COMPILE
    _NATIVE_EXEC(compiler(script.read_bytes(), str(script), 'exec', dont_inherit=True), scope)


if __name__ == '__main__':
    main()
