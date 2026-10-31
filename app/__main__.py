"""python -m app — bind to localhost and serve the lab UI."""

from __future__ import annotations

import os


def main() -> None:
    host = os.environ.get("LORALAB_HOST", "127.0.0.1")
    if host == "localhost":
        host = "127.0.0.1"
    if host != "127.0.0.1":
        raise SystemExit("LoraResultLab only binds to 127.0.0.1")
    port = int(os.environ.get("LORALAB_PORT", "8765"))
    import uvicorn

    from app.main import app

    print(f"LoraResultLab at http://{host}:{port}")
    print("ComfyUI submission is OFF unless you enable it in Settings and confirm each submit.")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
