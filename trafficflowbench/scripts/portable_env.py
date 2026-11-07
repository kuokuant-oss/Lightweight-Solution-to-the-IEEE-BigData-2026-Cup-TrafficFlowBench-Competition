"""Host-local environment selection; never carry compiled packages between OSes."""
# Interpreter selection helpers of the development tree's launchers. Nothing in the reproduction
# pipeline imports this module: run_reproduce.sh takes the interpreter from --python.
import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def environment_dir():
    """Returns the folder of this host's virtual environment.

    TFB_ENV_DIR overrides it; otherwise ~/.venvs/trafficflowbench-<os>-<machine>-py312.
    """
    override = os.environ.get('TFB_ENV_DIR')
    if override:
        return Path(override).expanduser().absolute()
    tag = platform.system().lower() + '-' + platform.machine().lower() + '-py312'
    return Path.home() / '.venvs' / ('trafficflowbench-' + tag)


def python_in(directory):
    """Returns the Python executable inside environment folder `directory`."""
    return directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def select_python():
    """Returns the interpreter to use.

    Order: TFB_PYTHON, then the host environment of environment_dir(), then (macOS only, without
    TFB_ENV_DIR) the repository's .venv.

    Raises:
        SystemExit: no environment exists for this host.
    """
    if os.environ.get('TFB_PYTHON'):
        return Path(os.environ['TFB_PYTHON']).expanduser()
    candidate = python_in(environment_dir())
    if candidate.is_file():
        return candidate
    # Preserve this Mac's working environment, but never reuse its link on Linux.
    legacy = REPO / '.venv/bin/python'
    if platform.system() == 'Darwin' and not os.environ.get('TFB_ENV_DIR') and legacy.is_file():
        return legacy
    raise SystemExit('No environment for this host. Run: sh bin/setup-env (Windows: bin\\setup-windows.cmd)')


def verify_python(python):
    """Checks that `python` runs Python 3.12 on this host's OS and architecture.

    Raises:
        SystemExit: the OS, architecture or Python version differs.
    """
    code = 'import platform,sys;print(platform.system());print(platform.machine());print("%s.%s"%sys.version_info[:2])'
    result = subprocess.check_output([str(python), '-I', '-c', code], text=True).splitlines()
    if result != [platform.system(), platform.machine(), '3.12']:
        raise SystemExit('Environment OS/architecture/Python mismatch; rebuild with bin/setup-env')
