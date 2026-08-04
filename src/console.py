"""Make stdout/stderr safe for the non-ASCII text these scripts print.

Every script prints box characters (✓ ⚠ ▲ ↔) and, for RMSC-V2 and PHATE, raw
Chinese and Persian label descriptions. On Windows the console default is often
a legacy code page (cp1252 / cp1256 / cp936), where a bare `print` of that text
raises UnicodeEncodeError and kills the run — after training, at the reporting
step, which is the worst possible moment.

Call `enable_utf8_output()` at the top of `main()`.
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """Switch stdout/stderr to UTF-8, replacing anything unencodable.

    `errors="replace"` means a terminal that genuinely cannot render a glyph
    prints a placeholder instead of crashing. No-op on Python < 3.7 streams or
    when the stream has already been redirected to something non-reconfigurable.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
