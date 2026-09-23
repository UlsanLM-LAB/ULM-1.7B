"""Run the ULM-1.7B HTTP/SSE inference server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ulm.inference.server import main  # noqa: E402

if __name__ == "__main__":
    main()
