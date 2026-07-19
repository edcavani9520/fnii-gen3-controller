#!/usr/bin/env python3
"""Forward Gen3-side commands to the canonical real-time deblur launcher."""

import argparse
import importlib.util
from pathlib import Path
import sys


DEBLUR_REPOSITORY_URL = (
    "https://github.com/edcavani9520/"
    "Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git"
)
CONTROLLER_ROOT = Path(__file__).resolve().parent
DEFAULT_DEBLUR_ROOT = Path(
    "../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin"
)


def load_deblur_launcher(deblur_root):
    """Load the canonical launcher without importing it at module import time."""
    deblur_root = Path(deblur_root).expanduser().resolve()
    source = deblur_root / "ws_inference_realtime_deblur.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Canonical real-time deblur launcher not found: {source}\n"
            f"Download it with:\n  git clone {DEBLUR_REPOSITORY_URL}\n"
            "Then select the checkout using --deblur-root."
        )

    root_text = str(deblur_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(
        "_canonical_realtime_deblur_ws", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)):
        raise ImportError(f"{source} does not define a callable main(argv)")
    return module


def _parse_forwarding_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--deblur-root",
        type=Path,
        default=DEFAULT_DEBLUR_ROOT,
    )
    known, forwarded = parser.parse_known_args(argv)
    return known.deblur_root, forwarded


def main(argv=None):
    deblur_root, forwarded = _parse_forwarding_args(argv)
    canonical = load_deblur_launcher(deblur_root)
    canonical.main(
        [
            *forwarded,
            "--controller-root",
            str(CONTROLLER_ROOT),
        ]
    )


if __name__ == "__main__":
    main()
