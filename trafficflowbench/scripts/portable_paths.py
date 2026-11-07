"""Explicit SSD relocation. Provenance files are never rewritten on read."""
import json
import os
import re
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
OLD_ROOTS = ('/Volumes/Crucial X9/trafficflow',
             '/Volumes/Crucial X9 1/trafficflow', '/Volumes/X9-2TB/trafficflow')
_INDEX = REPO / 'environment/portable-files.json'
_data = json.loads(_INDEX.read_text(encoding='utf-8')) if _INDEX.is_file() else {}
_renames = _data.get('renames', {})
_links = _data.get('links', {})
_link_prefixes = sorted(_links, key=len, reverse=True)


def safe_component(value):
    """Injective encoding for NEW artifact names (not IDs inside a dataset)."""
    value = str(value)
    value = re.sub(r'[%<>:"/\\|?*\x00-\x1f]', lambda m: '%%%02X' % ord(m[0]), value)
    while value.endswith((' ', '.')):
        value = value[:-1] + '%%%02X' % ord(value[-1])
    if re.match(r'^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)', value, re.I):
        value = '%%%02X' % ord(value[0]) + value[1:]
    return value or '%'


def _link_prefix(tail):
    parts = (tail or '').split('/')
    for count in range(len(parts), 0, -1):
        prefix = '/'.join(parts[:count])
        if prefix in _links:
            return prefix
    return None


def _tail(raw):
    raw = raw.replace('\\', '/')
    # Package edit (2026-11-07): test the run tree's own root first, so a run tree placed under
    # one of the old roots (e.g. /Volumes/X9-2TB/trafficflow/...) is not remapped onto itself.
    roots = (WORKSPACE.as_posix(), *OLD_ROOTS)
    for prefix in roots:
        if raw == prefix or raw.startswith(prefix + '/'):
            return raw[len(prefix):].lstrip('/')
    # The documented removable-drive layout; never match arbitrary other folders.
    match = re.match(r'^(?:[A-Za-z]:|/mnt/[A-Za-z])/trafficflow(?:/|$)', raw)
    if match:
        return raw[match.end():]
    if raw.startswith('tfb://repo/'):
        return REPO.name + '/' + raw[len('tfb://repo/'):]
    if raw.startswith('tfb://workspace/'):
        return raw[len('tfb://workspace/'):]
    return None


def resolve_string(value, *, workspace=None, follow_links=None):
    """Map old absolute roots and cache names; bytes/FDs retain their types."""
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return os.fsencode(resolve_string(os.fsdecode(value), workspace=workspace,
                                         follow_links=follow_links))
    raw = str(value) if isinstance(value, Path) else os.fspath(value)
    tail = _tail(raw)
    if tail is None:
        # Relative project paths also need renamed-cache resolution.
        if not Path(raw).is_absolute() and not re.match(r'^[A-Za-z]:', raw):
            absolute = Path(os.path.abspath(raw)).as_posix()
            candidate = _tail(absolute)
            if candidate in _renames or _link_prefix(candidate) is not None:
                tail = candidate
        if tail is None:
            return raw
    if '..' in PurePosixPath(tail).parts:
        raise ValueError('Parent traversal is not a relocation path')
    legacy_env = REPO.name + '/.venv'
    if os.environ.get('TFB_RUNTIME_ENV') and (tail == legacy_env or tail.startswith(legacy_env + '/')):
        return str(Path(os.environ['TFB_RUNTIME_ENV']) / tail[len(legacy_env):].lstrip('/'))
    # macOS understands its existing links; Windows/WSL need the recorded target.
    if follow_links is None:
        follow_links = os.environ.get('TFB_PORTABLE_LINKS') == '1' or os.name == 'nt'
        if not follow_links:
            import sys
            follow_links = sys.platform.startswith('linux')
    if follow_links:
        for _ in range(32):
            match = _link_prefix(tail)
            if match is None:
                break
            tail = _links[match] + tail[len(match):]
        else:
            raise ValueError('Portable link cycle')
    tail = _renames.get(tail, tail)
    return str(Path(workspace or WORKSPACE) / tail)


def rebase_text(value):
    # For frozen Python source literals and source fragments only. JSON unchanged.
    def literal_path(text):
        result = resolve_string(text)
        if text.endswith(('/', '\\')) and not result.endswith(('/', '\\')):
            result += '/'
        return result
    if (re.match(r'^(?:[A-Za-z]:[/\\]|/mnt/[A-Za-z]/)trafficflow(?:[/\\]|$)', value)
            and '\n' not in value):
        return literal_path(value)
    # Package edit (2026-11-07): one pass over all roots, with the run tree's own root listed
    # first, so text already under the run tree is never rebased a second time.
    _roots = (WORKSPACE.as_posix(), *OLD_ROOTS)
    value = re.sub('(?:' + '|'.join(re.escape(r) for r in _roots) + r')(?=$|[/\s\x22\x27])',
                   lambda _: WORKSPACE.as_posix(), value)
    if _tail(value) is not None and '\n' not in value:
        return literal_path(value)
    return value


def logical_name(value):
    """Recover an original filename when inspecting renamed historical caches."""
    raw = Path(value).absolute().as_posix()
    tail = _tail(raw)
    reverse = {v: k for k, v in _renames.items()}
    return Path(reverse.get(tail, tail or raw)).name
