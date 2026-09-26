"""Explicit setup-time acquisition of checksum-pinned Small weights.

Run python -m media_analysis.tools.vendor_beat_this /models. This tool is never
called from startup or inference. A failed transfer cannot replace valid weights.
"""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from media_analysis.features.beat_this import CHECKPOINT_SHA256, CHECKPOINT_URL


def vendor(destination: Path) -> Path:
    """Download at most the expected artifact size and atomically install it."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "beat-this-small0.ckpt"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == CHECKPOINT_SHA256:
        return target
    fd, name = tempfile.mkstemp(dir=destination, prefix=".beat-this-")
    try:
        with (
            os.fdopen(fd, "wb") as output,
            urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response,
        ):
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > 8_451_101:
                    raise ValueError("CHECKPOINT_SIZE_MISMATCH")
                output.write(chunk)
        data = Path(name).read_bytes()
        if len(data) != 8_451_101 or hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("CHECKPOINT_CHECKSUM_MISMATCH")
        os.chmod(name, 0o644)
        os.replace(name, target)
        return target
    finally:
        Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(vendor(parser.parse_args().destination))
