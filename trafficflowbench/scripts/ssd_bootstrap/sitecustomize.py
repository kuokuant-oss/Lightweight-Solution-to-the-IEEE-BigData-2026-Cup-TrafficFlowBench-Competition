"""Only the explicit tfb-python launcher enables frozen-source relocation."""
# Python imports sitecustomize at start-up from the first folder on sys.path that has one.
# bin/tfb-run (the launcher in this package) puts this folder first on PYTHONPATH and sets
# TFB_SSD_COMPAT=1, so every interpreter it starts (and every Python child process that inherits
# that environment) installs the path layer of ssd_runtime.py before any program code runs.
import os
if os.environ.get('TFB_SSD_COMPAT') == '1':
    from ssd_runtime import install
    install()
