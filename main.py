import uvicorn

from recorder_transcriber.api import app
from recorder_transcriber.core.settings import load_config
from recorder_transcriber.core.logger import setup_logging


def main() -> None:
    config = load_config()
    setup_logging(config.logging, config.paths.fs_dir)
    uvicorn.run(app, host="0.0.0.0", port=1643)


if __name__ == "__main__":
    main()
