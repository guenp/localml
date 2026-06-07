"""Reset local state.

Scaffold: brings the Docker Compose stack down and removes volumes, then back up. Phase 1
adds a faster in-place DB truncate + re-seed path.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    print("Tearing down stack and volumes ...")
    subprocess.run(["docker", "compose", "down", "-v"], check=False)
    print("Restarting stack ...")
    rc = subprocess.run(["docker", "compose", "up", "-d"], check=False).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
