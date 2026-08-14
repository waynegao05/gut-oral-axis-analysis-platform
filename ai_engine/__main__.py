from __future__ import annotations

import argparse
import os

from ai_engine.api import create_app
from ai_engine.runtime import validate_bind_host


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only AI Engine API.")
    parser.add_argument(
        "--host",
        default=os.getenv("GOA_ENGINE_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GOA_ENGINE_PORT", "8766")),
    )
    args = parser.parse_args()
    host = validate_bind_host(args.host)

    token = os.getenv("GOA_ENGINE_TOKEN")
    if not token or len(token) < 32:
        parser.error(
            "GOA_ENGINE_TOKEN must be set to a random value of at least 32 characters."
        )

    import uvicorn

    uvicorn.run(
        create_app(engine_token=token),
        host=host,
        port=args.port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
