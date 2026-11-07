"""Only the explicit tfb-python launcher enables frozen-source relocation."""
import os
if os.environ.get('TFB_SSD_COMPAT') == '1':
    from ssd_runtime import install
    install()
