"""Setup-only acquisition of Apache-2.0 Depth Anything V2 Small HF artifacts.

Use python -m media_analysis.tools.vendor_depth /models/depth-v1. Existing valid
files are reused. Downloads are size/checksum bounded and individually atomic;
startup verifies the complete set before declaring readiness. Never called by a
request or worker startup. Retain the upstream license with deployment records.
"""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from media_analysis.features.depth import FILES, MODEL, REVISION, verify_files


def vendor(folder: Path) -> None:
    """Install only the pinned artifact bytes; a failed transfer is not promoted."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        target = folder / name
        if (
            target.exists()
            and target.stat().st_size == size
            and hashlib.sha256(target.read_bytes()).hexdigest() == digest
        ):
            continue
        fd, temp = tempfile.mkstemp(dir=folder, prefix=".depth-")
        try:
            with (
                os.fdopen(fd, "wb") as out,
                urllib.request.urlopen(
                    f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}",
                    timeout=60,
                ) as response,
            ):
                total = 0
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("DEPTH_MODEL_SIZE_MISMATCH")
                    out.write(chunk)
            if total != size or hashlib.sha256(Path(temp).read_bytes()).hexdigest() != digest:
                raise ValueError("DEPTH_MODEL_CHECKSUM_MISMATCH")
            os.chmod(temp, 0o644)
            os.replace(temp, target)
        finally:
            Path(temp).unlink(missing_ok=True)
    verify_files(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    vendor(parser.parse_args().destination)
