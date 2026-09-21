"""Start the API with settings from config.yaml.

From Task[03]:
    python -m app.run
Then open http://127.0.0.1:8000/docs
"""

import uvicorn
from src.config import load_config


def main() -> None:
    cfg = load_config()
    api = cfg.get("api", {})
    uvicorn.run(
        "app.main:app",
        host=api.get("host", "0.0.0.0"),
        port=int(api.get("port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    main()
