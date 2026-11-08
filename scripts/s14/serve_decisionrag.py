"""Start the explicitly configured private CPU probability gateway."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute-local-cpu", action="store_true", required=True)
    args = parser.parse_args()
    from citequest.research_v3.decisionrag_gateway import GatewayConfig, create_app
    config = GatewayConfig.model_validate_json(args.config.read_bytes(), strict=True)
    import uvicorn
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1,
        access_log=False, limit_concurrency=4, backlog=1, timeout_keep_alive=2, timeout_graceful_shutdown=5,
        proxy_headers=False, server_header=False)


if __name__ == "__main__":
    main()
