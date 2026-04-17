from __future__ import annotations

import uvicorn

from conarrative.app import create_app
from conarrative.spaces import build_space_config


config = build_space_config()
app = create_app(config)


def main() -> None:
    uvicorn.run(app, host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    main()
