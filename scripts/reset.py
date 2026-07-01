"""Reset local state.

Default (fast) path: truncate every table in place and re-seed the default user + project —
handy between local runs without tearing down volumes. Pass ``--volumes`` to instead bring the
Docker Compose stack down with volumes and back up (full clean slate).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

from app.db import Base
from app.session import engine, init_db
from seed import main as seed


def reset_db() -> None:
    print("Dropping and recreating all tables ...")
    Base.metadata.drop_all(engine)
    init_db()
    seed()


def reset_volumes() -> None:
    print("Tearing down stack and volumes ...")
    subprocess.run(["docker", "compose", "down", "-v"], check=False)
    print("Restarting stack ...")
    rc = subprocess.run(["docker", "compose", "up", "-d"], check=False).returncode
    sys.exit(rc)


def main() -> None:
    if "--volumes" in sys.argv:
        reset_volumes()
    else:
        reset_db()


if __name__ == "__main__":
    main()
