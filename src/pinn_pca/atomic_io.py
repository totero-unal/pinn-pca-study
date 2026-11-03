"""Atomic JSON writes resilient to temporary Windows file sharing locks."""
import json
from pathlib import Path
import tempfile
import time
import warnings


def write_json(path, data, *, critical=True):
    """Retry locked replacements; progress-only failures must not stop training.

    Configuration, completion markers and scientific records remain critical.
    Never truncate an existing destination as a fallback for failed replacement.
    """
    path=Path(path)
    payload=json.dumps(data,indent=2,ensure_ascii=False)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(payload)
        for attempt in range(8):
            try:
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt==7:raise
                time.sleep(min(.05*2**attempt,.5))
    except OSError:
        if critical:raise
        warnings.warn(f'Could not update progress file {path}; training continues and the previous status may be stale.',RuntimeWarning,stacklevel=2)
        return False
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass
